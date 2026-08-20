# 部署手册（M4 — docker-compose；M5 — Kubernetes）

## M5：Kubernetes + Helm + Kueue 调度

### 组件与角色

| 组件 | 角色 |
| --- | --- |
| Helm chart `deploy/helm/minillm` | 一键渲染推理/训练/压测/队列全部清单 |
| vLLM Deployment + HPA | 常驻推理服务，按排队请求数弹性伸缩 |
| 训练/压测 Job | 一次性批任务，经 Kueue 队列调度（配额管理） |
| Kueue ClusterQueue/LocalQueue | GPU 配额、排队、优先级 |
| prometheus-adapter（外部依赖） | 把 vLLM `/metrics` 暴露为 HPA external 指标 |

### 前置条件

```bash
# 集群要求
- Kubernetes 1.28+，GPU 节点（nvidia-device-plugin 已装）
- GPU 节点打标: kubectl label node <node> gpu=true
- Kueue 已安装: kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/latest/download/manifests.yaml
- prometheus-adapter（HPA 外部指标）
- PVC: minillm-models（含微调产物）、minillm-data（训练数据）
```

### 部署推理服务

```bash
helm repo add minillm ./deploy/helm 2>/dev/null || true   # 或直接本地路径
helm upgrade --install minillm deploy/helm/minillm \
  --namespace minillm --create-namespace \
  --set storage.modelPVC=minillm-models \
  --set vllm.model=/models/runs/qwen25-1.5b-lora/export

# 验证
kubectl get deploy,svc,hpa -n minillm
kubectl exec -n minillm deploy/minillm-vllm -- curl -s localhost:8000/v1/models
```

### 提交训练/压测任务（Kueue 队列）

```bash
# 训练（经 LocalQueue minillm-training 排队，受 ClusterQueue GPU 配额约束）
helm upgrade --install minillm deploy/helm/minillm \
  --set training.enabled=true \
  --set training.config=configs/train/lora_qwen25_1p5b.yaml

kubectl get workloads -n minillm          # Kueue Workload 状态（admitted/pending）
kubectl get clusterqueue minillm-gpu -o yaml   # 配额使用情况

# 压测（三引擎对比）
helm upgrade --install minillm deploy/helm/minillm \
  --set bench.enabled=true \
  --set bench.config=configs/bench/matrix_qwen25.yaml
```

### HPA 弹性伸缩

```yaml
# values.yaml 中已预置：平均排队请求 > 20 时扩容（1 → 4 副本）
# HPA external 指标依赖 prometheus-adapter 配置：
#   - 从 vllm 服务抓取 vllm:num_requests_waiting
#   - 对外暴露为 external.metrics.k8s.io/vllm_num_requests_waiting
kubectl get hpa minillm-vllm -n minillm -w   # 观察扩缩容
```

### GPU 调度策略汇总

| 策略 | 实现 |
| --- | --- |
| GPU 资源声明 | `resources.limits.nvidia.com/gpu`（Deployment/Job 均声明） |
| 节点绑定 | nodeSelector `gpu=true` + tolerations |
| 批任务排队 | Kueue ClusterQueue 配额（cpu/memory/nvidia.com/gpu）→ LocalQueue 注解 |
| 服务弹性 | HPA external 指标（vLLM 排队数） |
| 张量并行 | `vllm.tensorParallelSize` = 副本内 GPU 数 |

### 本地验证（无集群）

```bash
make helm-tool        # 下载 helm 到 .tools/
make helm-validate    # lint + 全量渲染
make test             # 含 helm 渲染/引用完整性测试
```

---

## M4：docker-compose（开发/演示）

## M6：监控完善（告警 + Benchmark/训练结果入库）

### 数据流

```
minillm bench ──► Pushgateway(:9091) ──► Prometheus ──► Grafana "Benchmark 对比" 面板
minillm train ──► Pushgateway(:9091) ──► Prometheus ──► Grafana 训练面板
（config.push_gateway 或环境变量 MINILLM_PUSHGATEWAY 指定；未配置时自动跳过）
```

### 入库指标

| 指标 | 标签 | 来源 |
| --- | --- | --- |
| `minillm_bench_throughput_tps` | experiment/engine/model/concurrency/input/output | bench |
| `minillm_bench_ttft_p50/p99_seconds` | 同上 | bench |
| `minillm_bench_itl_p50_seconds` / `minillm_bench_e2e_p99_seconds` | 同上 | bench |
| `minillm_bench_success_rate` | 同上 | bench |
| `minillm_train_final_eval_loss` | experiment/model/mode | train |
| `minillm_train_tokens_per_second` | experiment/model/mode | train |

### 使用

```bash
# compose 环境（pushgateway 已内置）：
docker compose --env-file .env up -d pushgateway prometheus grafana
minillm bench --config configs/bench/smoke_local.yaml   # 自动推送（读 PUSHGATEWAY_URL）

# 裸机环境：
export MINILLM_PUSHGATEWAY=http://127.0.0.1:9091
minillm bench --config configs/bench/matrix_qwen25.yaml

# Grafana: miniLLM Benchmark 对比 面板（实验下拉过滤；吞吐/TTFT/ITL 柱状对比 + 训练曲线）
```

### 告警规则（6 条，Prometheus 预置）

服务不可达(critical) · GPU 显存 >90%(warn) · GPU 利用率 <5%(warn) · vLLM 排队 >50(warn) · TTFT-p99 >5s(warn) · abort 率 >10%(critical)

### 验证（无 Docker/集群）

```bash
make deploy-validate   # 校验 pushgateway 服务/采集/面板一致性
make test              # 含 Pushgateway 推送格式测试（本地捕获服务器）
```

---

# miniLLM Compose 架构（M4）

```
                 ┌─────────────┐
   客户端/压测 ──►│  端口映射区   │ 8000=vLLM · 30000=SGLang · 8001=serve-hf
                 └──────┬──────┘
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   ┌─────────┐    ┌──────────┐    ┌───────────┐
   │  vllm   │    │  sglang  │    │  serve-hf │   (GPU 服务经 docker-compose.gpu.yml 声明设备)
   └────┬────┘    └────┬─────┘    └─────┬─────┘
        └──────────────┴───────────────┘
        scrape ▼ (Prometheus 15s)
   ┌───────────┐     ┌────────┐     ┌─────────┐
   │ Prometheus │◄───►│ Grafana │     │  mlflow │
   │  +DCGM 指标 │     │ 面板预置 │     │ 实验追踪 │
   └───────────┘     └────────┘     └─────────┘
```

## 快速开始

### 演示模式（无 GPU：监控 + HF 推理 + MLflow）

```bash
cd deploy/compose
cp .env.example .env
docker compose --env-file .env up -d prometheus grafana serve-hf mlflow

# 验证
curl http://127.0.0.1:8001/v1/models          # HF OpenAI 兼容服务
open http://127.0.0.1:3000                    # Grafana（admin/admin，面板已预置）
open http://127.0.0.1:9090/targets             # Prometheus 抓取目标
```

### GPU 全栈（vLLM + SGLang + DCGM + 监控）

```bash
cd deploy/compose
cp .env.example .env
# 修改 .env: VLLM_MODEL/SGLANG_MODEL 指向实际微调产物
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --env-file .env --profile gpu up -d

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/runs/qwen25-1.5b-lora/export","messages":[{"role":"user","content":"hi"}]}'
```

### 训练任务（一次性容器）

```bash
cd deploy/compose
# CPU 训练（冒烟）：TRAIN_CONFIG=configs/train/smoke_scratch.yaml
docker compose --env-file .env --profile train run --rm train

# GPU 训练（LoRA Qwen2.5-1.5B）：
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --env-file .env --profile train run --rm train
# 指标自动入库 MLflow（http://127.0.0.1:5000）
```

## 服务与端口

| 服务 | 端口 | 说明 | 镜像 |
| --- | --- | --- | --- |
| vllm | 8000 | OpenAI 兼容推理（Paged Attention/Continuous Batching/KV Cache） | `vllm/vllm-openai:v0.12.0` |
| sglang | 30000 | 对比引擎 | `lmsysorg/sglang:latest` |
| serve-hf | 8001 | 平台 Transformers 服务（无 GPU 演示） | 自建 `Dockerfile.serve` |
| train | — | 一次性训练任务 | 自建 `Dockerfile.train` |
| mlflow | 5000 | 实验追踪 + 模型注册 | `ghcr.io/mlflow/mlflow` |
| prometheus | 9090 | 指标采集（15s） | `prom/prometheus:v2.53.0` |
| grafana | 3000 | 可视化（面板预置） | `grafana/grafana:11.1.0` |
| dcgm-exporter | 9400 | GPU 硬件指标 | `nvidia/dcgm-exporter:3.3.5` |

## 监控内容

- **Grafana 面板（自动预置）**：GPU 利用率/显存/温度、推理吞吐（tokens/s）、
  vLLM 运行/排队请求、TTFT p50/p99、KV Cache 命中率与使用率、请求速率
- **告警规则**（Prometheus）：服务不可达、GPU 显存 >90%、GPU 利用率异常、
  vLLM 排队 >50、TTFT-p99 >5s、abort 率 >10%
- 数据源：vLLM `/metrics`（`vllm:*`）、serve-hf `/metrics`（`minillm_*`）、DCGM（`DCGM_FI_DEV_*`）

## 关键设计

1. **双 compose 文件**：`docker-compose.yml`（基础）+ `docker-compose.gpu.yml`（override 设备声明）
   —— CPU 演示与 GPU 生产共享同一编排源；
2. **Profile 门控**：GPU 服务（vllm/sglang/dcgm）与训练任务按需启用，不污染 CPU 栈；
3. **同一镜像双环境**：Linux 下 pip 安装的 torch 自带 CUDA 运行库，训练/服务镜像
   在 GPU（+nvidia-container-toolkit）与 CPU 主机均可运行；
4. **健康检查**：服务均带 healthcheck（/health），供编排与 HPA 使用；
5. **无 Docker 环境验证**：`make deploy-validate` 静态校验全部编排/监控工件（见 tests/test_deploy_artifacts.py）。

## 环境变量（deploy/compose/.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HF_ENDPOINT` | hf-mirror.com | HuggingFace 镜像 |
| `VLLM_MODEL` / `SGLANG_MODEL` | runs/... | 模型路径（容器内 /models） |
| `VLLM_GPU_MEM_UTIL` | 0.85 | KV Cache 显存预算 |
| `VLLM_MAX_NUM_SEQS` | 256 | Continuous Batching 并发上限 |
| `VLLM_PREFIX_CACHING` | true | 前缀缓存 |
| `VLLM_GPU_COUNT` / `SGLANG_GPU_COUNT` | 1 | 张量并行卡数 |
| `GRAFANA_ADMIN_PASSWORD` | admin | Grafana 密码 |
