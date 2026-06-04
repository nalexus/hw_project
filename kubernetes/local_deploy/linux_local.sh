#!/usr/bin/env bash

# Bash entrypoint for Git Bash, WSL, or Linux shells.

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-trellis-classifier:latest}"
NODE_CONTAINER="${NODE_CONTAINER:-desktop-control-plane}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-trellis-classifier-api}"
SERVICE_NAME="${SERVICE_NAME:-trellis-classifier-api}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
PORT_FORWARD="${PORT_FORWARD:-false}"
SAVE_PORT_FORWARD_LOGS="${SAVE_PORT_FORWARD_LOGS:-false}"
PORT_FORWARD_LOG_DIR="${PORT_FORWARD_LOG_DIR:-.local/logs}"

# Returns success for common truthy environment variable values.
is_enabled() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

# Starts kubectl port-forward in the foreground unless log capture is enabled.
start_port_forward() {
  if ! is_enabled "${SAVE_PORT_FORWARD_LOGS}"; then
    echo "Starting Kubernetes port-forward on localhost:${LOCAL_PORT}"
    kubectl port-forward "service/${SERVICE_NAME}" "${LOCAL_PORT}:80"
    return
  fi

  mkdir -p "${PORT_FORWARD_LOG_DIR}"
  local out_log="${PORT_FORWARD_LOG_DIR}/port-forward.out.log"
  local err_log="${PORT_FORWARD_LOG_DIR}/port-forward.err.log"

  nohup kubectl port-forward "service/${SERVICE_NAME}" "${LOCAL_PORT}:80" >"${out_log}" 2>"${err_log}" &
  echo "Started background port-forward process: $!"
  echo "stdout log: ${out_log}"
  echo "stderr log: ${err_log}"
}

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Importing image into containerd"
docker save "${IMAGE_NAME}" | docker exec -i "${NODE_CONTAINER}" ctr -n k8s.io images import -

echo "Restarting Kubernetes deployment: ${DEPLOYMENT_NAME}"
kubectl rollout restart "deployment/${DEPLOYMENT_NAME}"
kubectl rollout status "deployment/${DEPLOYMENT_NAME}"

echo "Current pods:"
kubectl get pods

echo
if is_enabled "${PORT_FORWARD}"; then
  start_port_forward
else
  echo "If pods are Running, expose the API with:"
  echo "kubectl port-forward service/${SERVICE_NAME} ${LOCAL_PORT}:80"
  echo
  echo "Or run this script with PORT_FORWARD=true. Add SAVE_PORT_FORWARD_LOGS=true only when file logs are needed."
fi
