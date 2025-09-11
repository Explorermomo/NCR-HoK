import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from structure_encoder import MLPEncoder, GCNEncoder, GATEncoder
import os
from tqdm import tqdm
import time
import numpy as np

def merge_graph_datasets(graph_types, data_dir, merged_path):
    all_data = []
    for graph_type in graph_types:
        path = os.path.join(data_dir, f'{graph_type}_graphs.pt')
        data_list = torch.load(path, weights_only=False)
        all_data.extend(data_list)
    torch.save(all_data, merged_path)
    print(f"✅ 合并完成，保存为 {merged_path}，总样本数: {len(all_data)}")

def train(model, loader, criterion, optimizer, device, update=True):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc='Training', leave=False):
        data = data.to(device)
        if update:
            optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        if update:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Validation', leave=False):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Testing', leave=False):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_all_graphs(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    merged_path = os.path.join(args.data_dir, 'all_graphs.pt')
    if not os.path.exists(merged_path):
        merge_graph_datasets(['ER', 'SF', 'QSN', 'SW'], args.data_dir, merged_path)

    dataset = torch.load(merged_path)
    val_size = max(1, len(dataset) // 10)
    train_data = dataset[val_size:]
    val_data = dataset[:val_size]
    test_data = val_data

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    # 模型初始化
    if args.model == 'mlp':
        model = MLPEncoder().to(device)
    elif args.model == 'gcn':
        model = GCNEncoder().to(device)
    elif args.model == 'gat':
        model = GATEncoder().to(device)
    else:
        raise ValueError("Unsupported model type")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    save_path = os.path.join(args.save_dir, f'{args.model}_structure_encoder.pt')

    # 日志记录
    train_loss_list = []
    val_loss_list = []
    epoch_times = []

    # 🟡 Epoch 0：初始未训练模型的表现
    print(f"🟡 Epoch 0: Evaluating untrained model loss...")
    start_time = time.time()
    init_train_loss = train(model, train_loader, criterion, optimizer, device, update=False)
    init_val_loss = evaluate(model, val_loader, criterion, device)
    end_time = time.time()
    train_loss_list.append(init_train_loss)
    val_loss_list.append(init_val_loss)
    epoch_times.append(0.0)
    print(f"🔹 Epoch 0 | Train Loss: {init_train_loss:.4f} | Val Loss: {init_val_loss:.4f}")

    # ✅ 正式训练 Epoch 1 ~ N
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device, update=True)
        val_loss = evaluate(model, val_loader, criterion, device)
        end_time = time.time()
        elapsed = end_time - start_time

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        epoch_times.append(elapsed)

        print(f"[{args.model}] Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved at epoch {epoch} with val loss {val_loss:.4f}")

    # 保存训练日志
    np.save(os.path.join(args.save_dir, f'{args.model}_train_losses.npy'), np.array(train_loss_list))
    np.save(os.path.join(args.save_dir, f'{args.model}_val_losses.npy'), np.array(val_loss_list))
    np.save(os.path.join(args.save_dir, f'{args.model}_epoch_times.npy'), np.array(epoch_times))
    print(f"📝 日志保存于：{args.save_dir}")

    # 测试
    print(f"🔍 Testing best model...")
    model.load_state_dict(torch.load(save_path))
    test_loss = test(model, test_loader, criterion, device)
    print(f"🧪 Test L1 Loss: {test_loss:.4f}")

if __name__ == '__main__':
    class Args:
        data_dir = r'D:\BaiduNetdiskDownload\code_and_data\The_second_project'
        model = 'gat'
        epochs = 50
        lr = 1e-3
        save_dir = r'D:\BaiduNetdiskDownload\code_and_data\The_second_project'

    args = Args()

    # 🧪 先分别训练每类图（ER/SF/QSN/SW）
    for graph_type in ['ER', 'SF', 'QSN', 'SW']:


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = torch.load(os.path.join(args.data_dir, f'{graph_type}_graphs.pt'), weights_only=False)
        val_size = max(1, len(dataset) // 10)
        train_data = dataset[val_size:]
        val_data = dataset[:val_size]
        test_data = val_data

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1)
        test_loader = DataLoader(test_data, batch_size=1)

        # 模型选择
        if args.model == 'mlp':
            model = MLPEncoder().to(device)
        elif args.model == 'gcn':
            model = GCNEncoder().to(device)
        elif args.model == 'gat':
            model = GATEncoder().to(device)
        else:
            raise ValueError("Unsupported model type")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.L1Loss()

        # 保存路径
        save_path = os.path.join(args.save_dir, f'{graph_type}_{args.model}_structure_encoder.pt')

        train_loss_list, val_loss_list, epoch_times = [], [], []

        # Epoch 0
        print(f"🟡 Epoch 0: Evaluating untrained model...")
        init_train_loss = train(model, train_loader, criterion, optimizer, device, update=False)
        init_val_loss = evaluate(model, val_loader, criterion, device)
        train_loss_list.append(init_train_loss)
        val_loss_list.append(init_val_loss)
        epoch_times.append(0.0)
        print(f"🔹 Epoch 0 | Train Loss: {init_train_loss:.4f} | Val Loss: {init_val_loss:.4f}")

        best_val_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss = train(model, train_loader, criterion, optimizer, device, update=True)
            val_loss = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - start_time

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            epoch_times.append(elapsed)

            print(f"[{graph_type}] Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"✅ Model saved for {graph_type} at epoch {epoch} with val loss {val_loss:.4f}")

        # 保存日志
        np.save(os.path.join(args.save_dir, f'{graph_type}_{args.model}_train_losses.npy'), np.array(train_loss_list))
        np.save(os.path.join(args.save_dir, f'{graph_type}_{args.model}_val_losses.npy'), np.array(val_loss_list))
        np.save(os.path.join(args.save_dir, f'{graph_type}_{args.model}_epoch_times.npy'), np.array(epoch_times))

        # 测试
        print(f"🔍 Testing best {graph_type} model...")
        model.load_state_dict(torch.load(save_path))
        test_loss = test(model, test_loader, criterion, device)
        print(f"🧪 {graph_type} Test L1 Loss: {test_loss:.4f}")

    # ✅ 整合训练所有图
    print(f"\n=== 整合训练 {args.model} 模型 on ER+SF+QSN+SW 图 ===")
    train_all_graphs(args)
