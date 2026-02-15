import torch, time, numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import os

try:
    from thop import profile
except ImportError:
    print("thop not found, FLOPs calculation will be skipped. Install with `pip install thop`")
    profile = None
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2 
BATCH = 16
EPOCHS = 3

# Setup directories
os.makedirs("artifacts", exist_ok=True)
os.makedirs("results", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

data_dir = "data"
if not os.path.exists(os.path.join(data_dir, "train")) or not os.path.exists(os.path.join(data_dir, "val")):
    print(f"Error: Data directories not found at {data_dir}/train or {data_dir}/val")
    exit(1)

train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)

train_loader = DataLoader(train_ds,BATCH,shuffle=True)
val_loader = DataLoader(val_ds,BATCH)

models_dict = {
    "EfficientNet": models.efficientnet_b0(weights="DEFAULT"),
    "ResNet": models.resnet50(weights="DEFAULT"),
    "MobileNet": models.mobilenet_v2(weights="DEFAULT")
}

def modify(model):
    if hasattr(model,"classifier"):
        if isinstance(model.classifier, torch.nn.Sequential):
             model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,NUM_CLASSES)
        else:
             model.classifier=torch.nn.Linear(model.classifier.in_features,NUM_CLASSES)
    elif hasattr(model, "fc"):
        model.fc=torch.nn.Linear(model.fc.in_features,NUM_CLASSES)
    return model.to(device)

def topk(output,target,k=1):
    _,pred=output.topk(k,1,True,True)
    return pred.eq(target.view(-1,1)).sum().item()/target.size(0)

# Helper to convert numpy/torch types to json serializable
def json_serializable(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

print(f"Starting benchmark on device: {device}")

for name,model in models_dict.items():
    print(f"Training {name}...")
    model=modify(model)
    opt=torch.optim.Adam(model.parameters())
    loss_fn=torch.nn.CrossEntropyLoss()

    train_loss=[]
    for epoch in range(EPOCHS):
        model.train()
        ep_loss=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            out=model(x)
            loss=loss_fn(out,y)
            loss.backward()
            opt.step()
            ep_loss+=loss.item()
        train_loss.append(ep_loss)
        print(f"  Epoch {epoch+1}/{EPOCHS} Loss: {ep_loss:.4f}")

    model.eval()
    preds,targets=[],[]
    start=time.time()

    total_topk = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(device),y.to(device)
            o=model(x)
            preds.extend(o.argmax(1).cpu())
            targets.extend(y.cpu())
            total_topk += topk(o,y, k=1) * y.size(0) 

    infer_time=time.time()-start
    avg_topk = total_topk / len(val_ds)

    acc=accuracy_score(targets,preds)
    p,r,f,_=precision_recall_fscore_support(targets,preds,average="macro", zero_division=0)
    cm=confusion_matrix(targets,preds)

    flops = 0
    if profile:
        try:
            dummy=torch.randn(1,3,224,224).to(device)
            flops,_=profile(model,inputs=(dummy,),verbose=False)
            flops = flops/1e9
        except Exception:
            pass

    rob_acc = 0
    try:
        dummy=torch.randn(1,3,224,224).to(device)
        noise=torch.randn_like(dummy)*0.1
        rob=model(dummy+noise).argmax()==model(dummy).argmax()
        rob_acc = rob.item()
    except Exception:
        pass

    # Save Model Artifact
    artifact_path = os.path.join("artifacts", f"{name}.pth")
    torch.save(model.state_dict(), artifact_path)
    print(f"  Saved model to {artifact_path}")

    # Results Dictionary
    results = {
        "Model": name,
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1": f,
        "Confusion": cm.tolist(),
        "Top1": avg_topk,
        "TrainLoss": train_loss,
        "Generalization": train_loss[-1]-loss.item(),
        "Robustness": rob_acc,
        "FLOPs (G)": flops,
        "Inference Time (s)": infer_time
    }

    # Save Metrics JSON
    json_path = os.path.join("results", f"metrics_{name.lower()}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=json_serializable)
    print(f"  Saved metrics to {json_path}")
    
    print("-" * 30)

print("\nBenchmark Completed!")
