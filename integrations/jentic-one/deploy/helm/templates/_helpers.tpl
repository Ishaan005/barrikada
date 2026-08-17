{{- define "barrikade-jentic.name" -}}
{{- printf "%s-barrikade" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "barrikade-jentic.labels" -}}
app.kubernetes.io/part-of: jentic-one
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "barrikade-jentic.coreImage" -}}
{{- if .Values.barrikade.image.digest -}}
{{ printf "%s@%s" .Values.barrikade.image.repository .Values.barrikade.image.digest }}
{{- else -}}
{{ printf "%s:%s" .Values.barrikade.image.repository .Values.barrikade.image.tag }}
{{- end -}}
{{- end -}}

{{- define "barrikade-jentic.jenticImage" -}}
{{- if .Values.barrikade.enabled -}}
  {{- if .Values.jentic.barrikadeImage.digest -}}
{{ printf "%s@%s" .Values.jentic.barrikadeImage.repository .Values.jentic.barrikadeImage.digest }}
  {{- else -}}
{{ printf "%s:%s" .Values.jentic.barrikadeImage.repository .Values.jentic.barrikadeImage.tag }}
  {{- end -}}
{{- else -}}
{{ printf "%s@%s" .Values.jentic.stockImage.repository .Values.jentic.stockImage.digest }}
{{- end -}}
{{- end -}}

{{- define "barrikade-jentic.tokenSecret" -}}
{{- default (printf "%s-token" (include "barrikade-jentic.name" .)) .Values.barrikade.token.existingSecret -}}
{{- end -}}

{{- define "barrikade-jentic.databaseSecret" -}}
{{- default (printf "%s-database" (include "barrikade-jentic.name" .)) .Values.barrikade.database.existingSecret -}}
{{- end -}}
