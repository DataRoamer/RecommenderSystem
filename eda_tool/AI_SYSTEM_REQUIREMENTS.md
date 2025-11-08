# AI Features - System Requirements and Troubleshooting

## Overview

The EDA Tool includes optional AI-powered features using local LLM models via Ollama. These features provide intelligent data analysis assistance while maintaining 100% privacy (all processing happens locally).

## System Requirements

### Minimum Requirements for AI Features

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **RAM** | **16GB minimum** | 8GB systems will encounter memory errors |
| **Storage** | 5-10 GB free | For model downloads |
| **CPU** | Modern multi-core | For inference when GPU unavailable |
| **OS** | Windows 10/11, macOS, Linux | Via Ollama support |

### Recommended Specifications

| Component | Recommendation | Benefit |
|-----------|----------------|---------|
| **RAM** | 32GB+ | Smoother performance, faster responses |
| **GPU** | Dedicated GPU with 4GB+ VRAM | Significantly faster inference |
| **Storage** | SSD with 20GB+ free | Multiple models, faster loading |

## Common Issues and Solutions

### Issue 1: "Model requires more system memory than is currently available"

**Error Message:**
```
Error: API error: 500 - {"error":"model requires more system memory than is currently available unable to load full model on GPU"}
```

**Root Cause:**
- The system doesn't have enough free RAM to load the AI model
- Typically occurs on systems with 8GB RAM or less
- Windows OS + background applications + Ollama + model = insufficient memory

**Solutions:**

#### Option A: Free Up Memory (Temporary Fix)
1. **Close unnecessary applications:**
   - Close browser tabs you don't need
   - Close VS Code, IDEs, or development tools
   - Close Discord, Slack, or communication apps
   - Close video players or games

2. **Check Task Manager:**
   - Press `Ctrl + Shift + Esc`
   - Sort by "Memory" column
   - Close memory-intensive processes

3. **Restart and try again:**
   - Restart your computer
   - Only open essential applications
   - Start Ollama first
   - Then start the EDA Tool

#### Option B: Use a Smaller Model
The current implementation uses `phi3:mini` (2.3 GB), which is already one of the smallest models. However, if you still encounter issues:

1. Try `tinyllama` (~600 MB) - much smaller but lower quality
2. Note: This requires modifying the default model in AI Setup

#### Option C: Upgrade System RAM (Permanent Fix)
- Upgrade to 16GB+ RAM
- This is the most reliable long-term solution
- Enables use of better quality models

#### Option D: Test on Different Machine
- Use a computer with 16GB+ RAM
- Cloud instances or VMs with sufficient memory
- Development/workstation computers

### Issue 2: "Ollama is not installed"

**Symptoms:**
- Sidebar shows "AI: Not Available"
- AI Setup page shows "Ollama not installed"
- But Ollama IS installed on your system

**Root Cause:**
- Ollama application is installed but the service is not running
- The EDA Tool checks if Ollama API (port 11434) is accessible

**Solution:**

1. **Start Ollama Manually:**
   - Press Windows Key
   - Type "Ollama"
   - Click on Ollama application
   - Wait for llama icon to appear in system tray

2. **Verify Ollama is Running:**
   - Look for llama icon in system tray (bottom-right)
   - Refresh the EDA Tool browser page
   - Sidebar should show "AI: Available ✓"

3. **Start Ollama via Command Line (Alternative):**
   ```bash
   # Windows PowerShell
   Start-Process "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Ollama\ollama app.exe"

   # Or run Ollama serve directly
   ollama serve
   ```

### Issue 3: Slow AI Responses

**Symptoms:**
- AI Chat takes 30-60+ seconds to respond
- Progress spinner stays for a long time

**Causes & Solutions:**

1. **First Load (Normal):**
   - First query loads model into memory (30-60s)
   - Subsequent queries are faster
   - This is expected behavior

2. **Running on CPU instead of GPU:**
   - Check if you have a compatible GPU
   - Ensure GPU drivers are updated
   - Consider upgrading to a system with dedicated GPU

3. **System under heavy load:**
   - Close background applications
   - Check CPU usage in Task Manager
   - Ensure good system ventilation (thermal throttling)

## Memory Optimization Settings

The EDA Tool includes built-in memory optimizations:

```python
# Automatically applied settings:
- num_ctx: 512          # Minimal context window
- num_gpu: 0            # Force CPU mode (no GPU memory)
- num_thread: 1         # Single thread processing
- num_batch: 32         # Small batch size
- num_predict: 256      # Limited output length
```

These settings reduce memory usage but may affect:
- Response quality (shorter, less detailed)
- Response time (slower processing)
- Context awareness (limited conversation memory)

## Feature Availability by System

| RAM Size | AI Chat | AI Insights | Model Quality | Status |
|----------|---------|-------------|---------------|--------|
| 4-8 GB | ❌ | ❌ | N/A | Not supported |
| 8-12 GB | ⚠️ | ❌ | Very Low (tinyllama only) | Unreliable |
| 12-16 GB | ✅ | ⚠️ | Low-Medium (phi3:mini) | Basic functionality |
| 16-32 GB | ✅ | ✅ | Medium-High (llama3.1:8b) | Recommended |
| 32+ GB | ✅ | ✅ | High (larger models) | Optimal experience |

## Using the EDA Tool Without AI Features

**All core features work perfectly without AI:**
- ✅ Data upload and loading
- ✅ Data quality analysis
- ✅ Exploratory Data Analysis (EDA)
- ✅ Statistical analysis
- ✅ Visualizations
- ✅ Missing value analysis
- ✅ Correlation analysis
- ✅ Distribution analysis
- ✅ Report generation
- ✅ Code export

**AI features are optional enhancements:**
- AI Chat Assistant (requires 16GB+ RAM)
- AI-Generated Insights (requires 16GB+ RAM)
- Natural Language Queries (requires 16GB+ RAM)

## Testing AI Features

### Before Testing:
1. ✅ Check system has 16GB+ RAM
2. ✅ Install Ollama from https://ollama.ai
3. ✅ Start Ollama application
4. ✅ Download at least one model (phi3:mini recommended)
5. ✅ Close unnecessary applications to free memory

### Testing Checklist:
- [ ] AI Setup wizard completes successfully
- [ ] Sidebar shows "AI: Available ✓"
- [ ] phi3:mini model appears in installed models list
- [ ] Can set default model
- [ ] AI Chat loads without errors
- [ ] Can send a simple message ("Hello")
- [ ] Receives response within 60 seconds
- [ ] Subsequent messages respond faster

## Future Enhancements

Potential solutions for better 8GB RAM support:

1. **Model Quantization:**
   - Use more aggressively quantized models (Q2/Q3)
   - Trade quality for memory usage

2. **Cloud Integration (Optional):**
   - Provide option to use cloud-based LLM APIs
   - For users who can't run locally

3. **Streaming Responses:**
   - Stream responses token-by-token
   - Reduce memory footprint

4. **Selective Feature Loading:**
   - Only load model when AI features are actively used
   - Unload model when switching to other sections

## Support and Troubleshooting

### Check Ollama Status:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with installed models
```

### Check System Memory:
```bash
# Windows
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory

# macOS/Linux
free -h
```

### View Ollama Logs:
- Windows: Check Event Viewer or system tray icon
- macOS/Linux: Check ollama logs via `journalctl` or similar

## Contact & Feedback

If you encounter issues not covered in this document:
1. Check that Ollama is running (system tray icon)
2. Verify you have 16GB+ RAM available
3. Try restarting both Ollama and the EDA Tool
4. Review the troubleshooting steps above

## Version History

- **v1.0.0** (2025-11-06): Initial documentation
  - Documented 16GB RAM requirement
  - Added troubleshooting for memory errors
  - Added Ollama startup issues solutions
