{{/*
minillm fullname (truncated to 63 chars per DNS label limits)
*/}}
{{- define "minillm.fullname" -}}
{{- printf "%s-%s" .Release.Name "minillm" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "minillm.labels" -}}
app.kubernetes.io/name: minillm
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
GPU node selector (standardized across workloads)
*/}}
{{- define "minillm.gpuNodeSelector" -}}
{{- toYaml .Values.nodeSelector -}}
{{- end -}}
