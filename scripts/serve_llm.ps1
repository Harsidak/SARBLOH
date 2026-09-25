# Local OpenAI-compatible LLM server (llama.cpp) - the local stand-in for vLLM on Kaggle.
# Harness code talks to http://127.0.0.1:8080/v1 here and to vLLM's endpoint on Kaggle; nothing else changes.
#
#   .\scripts\serve_llm.ps1                      # default: Qwen3.5-4B Q4_K_M, text only, 16k context
#   .\scripts\serve_llm.ps1 -Vision              # also load the vision projector (image input)
#   .\scripts\serve_llm.ps1 -Model models\other\x.gguf -Ctx 8192
#
# Sized for an RTX 4050 Laptop (6 GB): the 3.0 GB Q4_K_M weights plus a q8_0 KV cache at 16k fit on the GPU.
# If you hit out-of-memory, lower -Ctx first; -Vision costs roughly another 0.7 GB.
param(
    [string]$Model = "models\qwen3.5-4b-gguf\Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    [string]$Mmproj = "models\qwen3.5-4b-gguf\mmproj-Qwen_Qwen3.5-4B-f16.gguf",
    [int]$Ctx = 16384,
    [int]$Port = 8080,
    [string]$Alias = "local",
    [switch]$Vision
)

$Root = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
if (-not (Test-Path $Server)) { throw "llama-server not found at $Server - see CLAUDE.md section 5" }
$ModelPath = Join-Path $Root $Model
if (-not (Test-Path $ModelPath)) { throw "model not found at $ModelPath - see CLAUDE.md section 5" }

$ServerArgs = @(
    "-m", $ModelPath,
    "--alias", $Alias,
    "--host", "127.0.0.1", "--port", $Port,
    "-c", $Ctx,
    "-ngl", "99",                         # all layers on the GPU
    "-fa", "on",                          # flash attention
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--jinja"                             # use the model's own chat template (tool calls, thinking)
)
if ($Vision) { $ServerArgs += @("--mmproj", (Join-Path $Root $Mmproj)) }

Write-Host "llama-server $($ServerArgs -join ' ')"
& $Server @ServerArgs
