# Kubernetes 学习资料（面向 AI 基础设施岗）

> 定位：不是从零的 K8s 教程，而是**针对"GPU 集群调度 / AI 负载"方向**的知识梳理 + 面试高频 + 实操路径。
> 配套：本项目 `deploy/helm/minillm/` 是现成的 Helm chart 实例，`docs/deployment.md` 是实战文档。

---

## 一、核心概念速览（必会，30 分钟）

| 概念 | 作用 | 一句话 |
| --- | --- | --- |
| Pod | 最小调度单元 | 一个或多个容器共享网络/存储 |
| Deployment | 无状态应用副本管理 | 滚动更新、回滚、副本数 |
| StatefulSet | 有状态应用 | 稳定网络标识 + 持久存储（如训练数据节点） |
| Job / CronJob | 一次性/定时任务 | **训练、压测这类批任务就用它** |
| Service | 稳定访问入口 | 负载均衡 + DNS |
| ConfigMap / Secret | 配置注入 | 配置与镜像分离 |
| Namespace | 资源隔离 | 逻辑分区 |
| Ingress | 七层路由 | HTTP 入口 |

**一句话记忆**：`Deployment 管常驻服务，Job 管批任务，Service 管访问，ConfigMap/Secret 管配置`。

---

## 二、调度器（AI 集群的核心）

### 调度流程
`Pod 创建 → 过滤（filter，硬性条件）→ 打分（score，择优）→ 绑定（bind，落到节点）`

### 关键机制
| 机制 | 作用 | 示例 |
| --- | --- | --- |
| nodeSelector | 节点标签硬匹配 | `gpu: "true"`（本项目 helm 就是打这个标签） |
| nodeAffinity | 软/硬亲和 | 优先调度到某类节点 |
| taint / toleration | 节点"排斥" + Pod"容忍" | GPU 节点打 taint，只有带 toleration 的 AI Pod 能上 |
| resources.requests/limits | 资源申请/上限 | `requests` 决定调度，`limits` 决定隔离 |
| QoS | 服务质量分级 | Guaranteed / Burstable / BestEffort |

### GPU 调度（重点）
- **Device Plugin**：让 K8s 识别 GPU，`resources.limits: nvidia.com/gpu: 1` 声明
- **GPU 共享**：MIG（物理切分，如 A100 切 7 份）/ vGPU / 时间片（MPS）
- **拓扑感知**：GPU 间 NVLink 距离影响 TP 通信，需 topology manager 感知 NUMA
- **本项目对照**：`deploy/helm/minillm/templates/inference-vllm.yaml` 里 `nvidia.com/gpu` 声明 + `nodeSelector: gpu=true`

---

## 三、AI 负载编排

### 批任务队列调度（Kueue / Volcano）
- 原生 K8s 对"资源够就调度、不够就 Pending"——**没有排队秩序和配额**
- **Kueue**：ClusterQueue（集群配额 nominalQuota）+ LocalQueue（命名空间入口）+ Workload 准入
  - 本项目已实现：训练/压测 Job 通过 `kueue.x-k8s.io/queue-name` 注解接入
- **Volcano**（另一主流）：gang scheduling（一组 Pod 同生共死，适合分布式训练多卡 Pod 必须同时调度）

### 弹性伸缩
| 机制 | 依据 | 适用 |
| --- | --- | --- |
| HPA | 资源/自定义指标 | 无状态推理服务（本项目用 vLLM 排队数扩缩） |
| Cluster Autoscaler / Karpenter | 节点池扩容 | 整机扩缩容 |

### 本项目实战对照（`deploy/helm/minillm/`）
- `inference-vllm.yaml`：Deployment + Service + **HPA（external 指标 vllm 排队数）**
- `training-job.yaml` / `bench-job.yaml`：Job + Kueue 队列注解
- `queue.yaml`：ClusterQueue + LocalQueue

---

## 四、存储与网络

| 主题 | 要点 |
| --- | --- |
| 存储 | PVC/PV、StorageClass；**模型权重**通常放共享存储（NFS/对象存储），训练日志/checkpoint 也要持久化 |
| 网络 | CNI（Calico/Cilium）；**AI 集群关键在 RDMA/InfiniBand**，跨机分布式训练通信对网络极敏感 |
| 容器运行时 | Docker → containerd；GPU 容器需 **nvidia-container-toolkit** |

---

## 五、可观测性

| 主题 | 要点 |
| --- | --- |
| 指标 | Prometheus + **DCGM Exporter**（GPU 利用率/显存/温度） |
| 日志 | 容器日志 → Loki / ELK |
| 面板 | Grafana（本项目已预置 GPU + 推理面板） |
| 告警 | 显存水位、服务不可达、排队积压（本项目 6 条规则） |

---

## 六、面试高频问答（可背诵）

### Q1. 说说 K8s 的调度过程？
过滤（nodeSelector/taint/资源够不够）→ 打分（亲和、资源均衡）→ 绑定。GPU 场景加 device plugin 声明 `nvidia.com/gpu`，用 taint/toleration 把 AI Pod 圈到 GPU 节点。

### Q2. Deployment 和 Job 区别？训练任务用什么？
Deployment 管**常驻**服务（滚动更新、副本恢复）；Job 管**一次性**任务（跑完退出、可重试）。训练/压测是批任务 → Job（+ Kueue 队列管理 GPU 配额）。

### Q3. requests 和 limits 的区别？
`requests` 是调度依据（保证量），`limits` 是上限（防超用）。只设 limits 不设 requests 时默认相等。GPU 上 requests==limits 才能精确隔离。

### Q4. 怎么保证 GPU 节点只跑 AI 负载？
节点打 taint（如 `nvidia.com/gpu:NoSchedule`），AI Pod 加对应 toleration；再结合 nodeSelector 或 device plugin 的扩展资源。

### Q5. 多个训练任务争抢 GPU 怎么办？
用队列调度：**Kueue ClusterQueue 设配额**（比如 2 卡），任务进 LocalQueue 排队，按配额/优先级/抢占准入；或 **Volcano gang scheduling** 保证多卡任务同批调度。

### Q6. 推理服务怎么自动扩容？
HPA 用**业务指标**（如 vLLM 的排队请求数 `vllm:num_requests_waiting`，经 prometheus-adapter 暴露成 external 指标），排队超阈值就扩副本；比 CPU 指标更贴近真实拥塞。

### Q7. 说说你项目里的 K8s 落地？
（结合本项目）Helm chart 一键渲染：vLLM Deployment + Service + HPA、训练/压测 Job 走 Kueue 队列、GPU 节点 nodeSelector + tolerations、PVC 挂模型。`helm lint` + 全量渲染校验，`helm template` 验证 12 个资源。

---

## 七、学习路径与实操清单（按顺序）

1. **上手**（1~2 天）：minikube / kind 起单节点，跑 Deployment + Service + `kubectl get/describe/logs/exec`
2. **调度**（1 天）：nodeSelector / affinity / taint-toleration / requests-limits，观察 `kubectl describe pod` 的 Events
3. **批任务**（1 天）：Job/CronJob，跑一个训练 Job，看 Pod 生命周期
4. **GPU**（1~2 天）：装 device plugin + `nvidia.com/gpu` 声明；装 DCGM exporter 看 GPU 指标
5. **队列调度**（1~2 天）：装 Kueue，建 ClusterQueue/LocalQueue，提交 Job 看排队/准入
6. **对照本项目**（半天）：`helm template demo deploy/helm/minillm` 逐资源读一遍 + `docs/deployment.md`

**关键原则**：K8s 别只背概念，一定要 `kubectl describe` 看真实 Events——AI 岗面试官爱追问"Pod 一直 Pending 你怎么查"，答案就是看 Events + 资源/污点/配额逐项排查。
