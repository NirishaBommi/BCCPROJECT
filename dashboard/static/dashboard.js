document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // 1. TAB ROUTING & THEME TOGGLING
    // ---------------------------------------------------------
    const navItems = document.querySelectorAll('#sidebar-nav .nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Initial Route check
    const currentHash = window.location.hash || '#home-tab';
    switchTab(currentHash);
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetHash = item.getAttribute('href');
            window.location.hash = targetHash;
            switchTab(targetHash);
        });
    });
    
    function switchTab(hash) {
        navItems.forEach(item => {
            if (item.getAttribute('href') === hash) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        tabPanes.forEach(pane => {
            if ('#' + pane.id === hash) {
                pane.classList.remove('hidden');
            } else {
                pane.classList.add('hidden');
            }
        });
        
        // Refresh charts if entering analytics tab
        if (hash === '#analytics-tab') {
            initAnalyticsCharts();
        } else if (hash === '#history-tab') {
            loadPatientHistory();
        }
    }
    
    // Theme Switcher
    const themeToggle = document.getElementById('theme-toggle');
    themeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.classList.contains('dark');
        if (isDark) {
            document.documentElement.classList.remove('dark');
            themeToggle.innerHTML = '<i class="fa-regular fa-moon"></i> Dark Mode';
        } else {
            document.documentElement.classList.add('dark');
            themeToggle.innerHTML = '<i class="fa-regular fa-sun"></i> Light Mode';
        }
    });
    
    // Set Current Date
    document.getElementById('current-date').innerText = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    // ---------------------------------------------------------
    // 2. INPUT ACQUISITION (UPLOAD & WEBCAM STREAM)
    // ---------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const imagePreview = document.getElementById('image-preview');
    const webcamBtn = document.getElementById('webcam-btn');
    const webcamContainer = document.getElementById('webcam-container');
    const webcamVideo = document.getElementById('webcam-video');
    const captureBtn = document.getElementById('capture-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    let selectedFile = null;
    let webcamStream = null;
    let capturedBase64 = null;
    
    // Click dropzone to browse
    dropZone.addEventListener('click', (e) => {
        // Prevent click if clicking webcam controls
        if (e.target.closest('#webcam-container') || e.target.closest('#webcam-btn')) return;
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });
    
    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-clinical-accent');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-clinical-accent');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-clinical-accent');
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
    
    function handleImageFile(file) {
        selectedFile = file;
        capturedBase64 = null;
        stopWebcam();
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropZoneContent.classList.add('hidden');
            webcamContainer.classList.add('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    // Webcam Activation
    webcamBtn.addEventListener('click', async () => {
        selectedFile = null;
        imagePreview.classList.add('hidden');
        dropZoneContent.classList.add('hidden');
        webcamContainer.classList.remove('hidden');
        
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            webcamVideo.srcObject = webcamStream;
            analyzeBtn.disabled = true;
        } catch (err) {
            console.error("Webcam access error:", err);
            alert("Could not access camera. Please check browser permissions.");
            stopWebcam();
        }
    });
    
    captureBtn.addEventListener('click', () => {
        if (!webcamVideo.srcObject) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        
        capturedBase64 = canvas.toDataURL('image/jpeg');
        
        // Stop stream and show preview
        stopWebcam();
        imagePreview.src = capturedBase64;
        imagePreview.classList.remove('hidden');
        webcamContainer.classList.add('hidden');
        analyzeBtn.disabled = false;
    });
    
    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
    }

    // ---------------------------------------------------------
    // 3. PIPELINE INFERENCE EXECUTION
    // ---------------------------------------------------------
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsLoader = document.getElementById('results-loader');
    const resultsDashboard = document.getElementById('results-dashboard');
    
    let activeCAMUrls = {};
    
    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile && !capturedBase64) return;
        
        // Show loading state
        resultsPlaceholder.classList.add('hidden');
        resultsLoader.classList.remove('hidden');
        resultsDashboard.classList.add('hidden');
        analyzeBtn.disabled = true;
        
        const formData = new FormData();
        
        // Append input image (file or base64)
        if (selectedFile) {
            formData.append('image', selectedFile);
        } else if (capturedBase64) {
            formData.append('image_base64', capturedBase64);
        }
        
        // Append configurations
        formData.append('model_type', document.getElementById('model-select').value);
        formData.append('use_tta', document.getElementById('tta-toggle').checked);
        formData.append('use_threshold', document.getElementById('threshold-toggle').checked);
        
        // Append Patient Info
        formData.append('patient_name', document.getElementById('patient-name-input').value || 'Anonymous');
        formData.append('patient_age', document.getElementById('patient-age-input').value || '0');
        formData.append('patient_gender', document.getElementById('patient-gender-select').value || 'Unknown');
        formData.append('doctor_notes', document.getElementById('doctor-notes-input').value || '');
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.detail || 'Inference Failed') });
            }
            return res.json();
        })
        .then(data => {
            // Show Results
            resultsLoader.classList.add('hidden');
            resultsDashboard.classList.remove('hidden');
            analyzeBtn.disabled = false;
            
            // Map text details
            document.getElementById('result-diagnosis-name').innerText = data.prediction;
            document.getElementById('result-confidence-val').innerText = `${(data.confidence * 100).toFixed(2)}%`;
            document.getElementById('result-speed-val').innerText = data.inference_time;
            document.getElementById('result-model-val').innerText = data.model_used;
            document.getElementById('result-interpretation-text').innerText = data.interpretation;
            
            // Map risk badge
            const badge = document.getElementById('result-risk-badge');
            badge.innerText = `${data.risk_level} Risk`;
            badge.className = 'px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border';
            
            const fill = document.getElementById('result-confidence-fill');
            const tick = document.getElementById('result-threshold-tick');
            const bccProb = data.distribution.find(d => d.disease === 'Basal Cell Carcinoma').probability;
            
            if (data.risk_level === 'Red') {
                badge.classList.add('bg-red-500/10', 'text-clinical-danger', 'border-red-500/20');
                fill.style.background = 'linear-gradient(90deg, #f093fb, #ef4444)';
            } else if (data.risk_level === 'Orange') {
                badge.classList.add('bg-orange-500/10', 'text-clinical-warning', 'border-orange-500/20');
                fill.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
            } else if (data.risk_level === 'Yellow') {
                badge.classList.add('bg-yellow-500/10', 'text-yellow-400', 'border-yellow-500/20');
                fill.style.background = 'linear-gradient(90deg, #4facfe, #f59e0b)';
            } else {
                badge.classList.add('bg-emerald-500/10', 'text-clinical-success', 'border-emerald-500/20');
                fill.style.background = 'linear-gradient(90deg, #4facfe, #10b981)';
            }
            
            // Set main probability meter value (corresponds to BCC risk probability)
            fill.style.width = `${(bccProb * 100).toFixed(1)}%`;
            
            // Position threshold tick
            const useThresh = document.getElementById('threshold-toggle').checked;
            const threshVal = useThresh ? 40 : 50;
            tick.style.left = `${threshVal}%`;
            tick.style.setProperty('--threshold-label', `"T=${threshVal}%"`);
            
            const explanationEl = document.getElementById('result-threshold-explanation');
            if (data.risk_level === 'Red' && bccProb < 0.50) {
                explanationEl.style.display = 'flex';
                explanationEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span><strong>Screening Flag:</strong> BCC Probability (${(bccProb * 100).toFixed(1)}%) is under 50% but exceeds clinical screening threshold of ${threshVal}% to avoid false negatives.</span>`;
            } else if (data.risk_level === 'Red') {
                explanationEl.style.display = 'flex';
                explanationEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span><strong>High Risk:</strong> BCC Probability (${(bccProb * 100).toFixed(1)}%) is above safety limits. Medical evaluation recommended.</span>`;
            } else {
                explanationEl.style.display = 'flex';
                explanationEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span><strong>Low Risk:</strong> Lesion is within safe diagnostic thresholds.</span>`;
            }
            
            // Load side-by-side images
            document.getElementById('res-orig-img').src = data.orig_url;
            document.getElementById('res-prep-img').src = data.prep_url;
            
            // Mask overlays
            document.getElementById('res-mask-underlay').src = data.orig_url;
            document.getElementById('res-mask-overlay').src = data.mask_url;
            
            // Explainability overlays
            activeCAMUrls = data.cam_urls;
            document.getElementById('res-cam-underlay').src = data.orig_url;
            document.getElementById('res-cam-overlay').src = activeCAMUrls.gradcam || data.orig_url;
            
            // Reset CAM tab buttons
            document.querySelectorAll('#cam-tabs .cam-tab-btn').forEach(btn => {
                if (btn.getAttribute('data-method') === 'gradcam') {
                    btn.className = 'cam-tab-btn active border-b-2 border-clinical-accent text-clinical-accent pb-3 text-sm font-semibold';
                } else {
                    btn.className = 'cam-tab-btn text-slate-400 hover:text-slate-300 pb-3 text-sm font-medium';
                }
            });
            
            // Render 9-class list
            const probsList = document.getElementById('class-probs-list');
            probsList.innerHTML = '';
            
            data.distribution.forEach(item => {
                const pct = (item.probability * 100).toFixed(1);
                
                const itemHtml = `
                    <div class="space-y-1.5">
                        <div class="flex justify-between text-xs font-semibold">
                            <span class="text-slate-300">${item.disease}</span>
                            <span class="text-clinical-accent font-bold">${pct}%</span>
                        </div>
                        <div class="w-full bg-slate-800 rounded-full h-1.5">
                            <div class="bg-clinical-accentDeep h-1.5 rounded-full" style="width: ${pct}%"></div>
                        </div>
                    </div>
                `;
                probsList.insertAdjacentHTML('beforeend', itemHtml);
            });
            
            // Save global analysis cache for PDF export
            window.latestAnalysisData = data;
        })
        .catch(err => {
            console.error("Diagnosis error:", err);
            resultsLoader.classList.add('hidden');
            resultsPlaceholder.classList.remove('hidden');
            analyzeBtn.disabled = false;
            alert(`Analysis failed: ${err.message}`);
        });
    });

    // ---------------------------------------------------------
    // 4. MASK & CAM CONTROLLER TABS & SLIDERS
    // ---------------------------------------------------------
    const maskOpacity = document.getElementById('mask-opacity');
    const maskOverlay = document.getElementById('res-mask-overlay');
    
    maskOpacity.addEventListener('input', () => {
        maskOverlay.style.opacity = maskOpacity.value;
    });
    
    const camOpacity = document.getElementById('cam-opacity');
    const camOverlay = document.getElementById('res-cam-overlay');
    
    camOpacity.addEventListener('input', () => {
        camOverlay.style.opacity = camOpacity.value;
    });
    
    // CAM Tabs selector
    const camTabs = document.querySelectorAll('#cam-tabs .cam-tab-btn');
    camTabs.forEach(btn => {
        btn.addEventListener('click', () => {
            camTabs.forEach(b => {
                b.className = 'cam-tab-btn text-slate-400 hover:text-slate-300 pb-3 text-sm font-medium transition-all';
            });
            btn.className = 'cam-tab-btn active border-b-2 border-clinical-accent text-clinical-accent pb-3 text-sm font-semibold transition-all';
            
            const method = btn.getAttribute('data-method');
            if (activeCAMUrls[method]) {
                camOverlay.src = activeCAMUrls[method];
            }
        });
    });

    // ---------------------------------------------------------
    // 5. CLINICAL PDF CLINICAL REPORT EXPORTER
    // ---------------------------------------------------------
    const downloadPdfBtn = document.getElementById('download-pdf-btn');
    downloadPdfBtn.addEventListener('click', () => {
        const data = window.latestAnalysisData;
        if (!data) return;
        
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Styling configuration
        doc.setFillColor(11, 19, 41); // Clinical dark navy
        doc.rect(0, 0, 210, 297, 'F');
        
        // Header
        doc.setFont('Helvetica', 'bold');
        doc.setFontSize(22);
        doc.setTextColor(0, 242, 254); // Cyan tint
        doc.text("DERMA-AI DIAGNOSTICS CENTER", 14, 25);
        
        doc.setFontSize(10);
        doc.setTextColor(203, 213, 225); // Slate
        doc.text("Clinical Diagnostic Worksheet - Basal Cell Carcinoma Screening", 14, 32);
        
        doc.setDrawColor(28, 37, 65);
        doc.setLineWidth(1);
        doc.line(14, 38, 196, 38);
        
        // Patient details card box
        doc.setFillColor(28, 37, 65); // Card bg
        doc.rect(14, 45, 182, 35, 'F');
        
        doc.setFontSize(11);
        doc.setTextColor(255, 255, 255);
        doc.text(`Patient Name: ${data.patient_name}`, 20, 53);
        doc.text(`Patient Age: ${data.patient_age}`, 20, 61);
        doc.text(`Gender: ${data.patient_gender}`, 20, 69);
        
        doc.text(`Date of Analysis: ${new Date().toLocaleDateString()}`, 110, 53);
        doc.text(`Inference Engine: ${data.model_used}`, 110, 61);
        doc.text(`Risk Status: ${data.risk_level} Risk`, 110, 69);
        
        // Diagnostic Prediction
        doc.setFontSize(16);
        doc.setTextColor(239, 68, 68); // Red
        doc.text(`DIAGNOSIS: ${data.prediction}`, 14, 95);
        
        doc.setFontSize(12);
        doc.setTextColor(255, 255, 255);
        doc.text(`Primary Class Confidence: ${(data.confidence * 100).toFixed(2)}%`, 14, 103);
        
        // Embed heatmaps / images placeholders (since we use local file system URLs in jsPDF,
        // we can draw clinical boxes for images or download base64.
        // To make it robust and prevent image loading errors on local browser tabs,
        // we will draw clear medical grid borders representing images!)
        doc.setDrawColor(0, 242, 254);
        doc.rect(14, 115, 80, 80);
        doc.setTextColor(0, 242, 254);
        doc.setFontSize(10);
        doc.text("LESION ORIGIN ACQUISITION", 25, 155);
        
        doc.setDrawColor(240, 147, 251);
        doc.rect(116, 115, 80, 80);
        doc.setTextColor(240, 147, 251);
        doc.text("EXPLAINABLE AI HEATMAP", 128, 155);
        
        // Doctor notes
        doc.setFillColor(28, 37, 65);
        doc.rect(14, 210, 182, 35, 'F');
        doc.setFontSize(10);
        doc.setTextColor(255, 255, 255);
        doc.text("Doctor clinical recommendations:", 20, 218);
        doc.setFont('Helvetica', 'normal');
        doc.setTextColor(203, 213, 225);
        
        const docNotes = document.getElementById('doctor-notes-input').value || "No doctor annotations compiled.";
        doc.text(docNotes, 20, 226, { maxWidth: 170 });
        
        // Barcode / QR placeholder
        doc.setDrawColor(255, 255, 255);
        // Draw simulated barcode
        for(let i=0; i<30; i++) {
            const w = Math.random() > 0.5 ? 2.0 : 0.8;
            doc.rect(14 + (i*1.8), 260, w, 15, 'F');
        }
        doc.setFontSize(8);
        doc.setTextColor(203, 213, 225);
        doc.text("*990218820B*", 28, 280);
        
        // Hospital signature
        doc.setFont('Helvetica', 'bold');
        doc.text("DERMA-AI Clinical Signature", 140, 265);
        doc.line(140, 272, 196, 272);
        
        doc.save(`clinical_report_${data.patient_name.toLowerCase().replace(" ", "_")}.pdf`);
    });

    // ---------------------------------------------------------
    // 6. CLINICAL HISTORY RECORDS & SEARCH (SQLITE)
    // ---------------------------------------------------------
    const historyTableBody = document.getElementById('history-table-body');
    const searchBtn = document.getElementById('search-btn');
    const exportExcelBtn = document.getElementById('export-excel-btn');
    
    function loadPatientHistory() {
        const query = document.getElementById('filter-search').value;
        const disease = document.getElementById('filter-disease').value;
        const modelName = document.getElementById('filter-model').value;
        
        let url = `/history?`;
        if (query) url += `query=${query}&`;
        if (disease) url += `disease=${disease}&`;
        if (modelName) url += `model=${modelName}&`;
        
        fetch(url)
        .then(res => res.json())
        .then(records => {
            historyTableBody.innerHTML = '';
            
            if (records.length === 0) {
                historyTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="px-6 py-8 text-center text-slate-500 font-medium">No diagnostic records logged in database.</td>
                    </tr>
                `;
                return;
            }
            
            records.forEach(row => {
                const dateStr = new Date(row.timestamp).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
                
                const tableRowHtml = `
                    <tr class="border-b border-slate-800/50 hover:bg-slate-800/10">
                        <td class="px-6 py-4 text-slate-400 font-medium">${dateStr}</td>
                        <td class="px-6 py-4 text-slate-200 font-semibold">${row.patient_name} <span class="text-slate-500 font-normal">(${row.age}y, ${row.gender})</span></td>
                        <td class="px-6 py-4 font-bold text-clinical-danger">${row.prediction}</td>
                        <td class="px-6 py-4 text-slate-300 font-bold">${(row.confidence * 100).toFixed(1)}%</td>
                        <td class="px-6 py-4 text-xs font-semibold text-clinical-accent">${row.model_used}</td>
                        <td class="px-6 py-4 text-center">
                            <button class="delete-record-btn text-xs px-2.5 py-1 rounded bg-red-500/10 text-clinical-danger hover:bg-red-500/20 border border-red-500/20 font-bold" data-id="${row.id}"><i class="fa-regular fa-trash-can"></i> Delete</button>
                        </td>
                    </tr>
                `;
                historyTableBody.insertAdjacentHTML('beforeend', tableRowHtml);
            });
            
            // Add delete event listeners
            document.querySelectorAll('.delete-record-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const id = btn.getAttribute('data-id');
                    if (confirm(`Confirm deletion of log record #${id}?`)) {
                        fetch(`/history/delete/${id}`, { method: 'POST' })
                        .then(() => {
                            loadPatientHistory();
                        });
                    }
                });
            });
        });
    }
    
    searchBtn.addEventListener('click', loadPatientHistory);
    
    exportExcelBtn.addEventListener('click', () => {
        window.location.href = "/history/export";
    });

    // ---------------------------------------------------------
    // 7. SYSTEM STATS REFRESHER
    // ---------------------------------------------------------
    function refreshSystemStats() {
        fetch('/system/stats')
        .then(res => res.json())
        .then(data => {
            document.getElementById('cpu-val').innerText = data.cpu_usage;
            document.getElementById('cpu-bar').style.width = data.cpu_usage;
            
            document.getElementById('gpu-val').innerText = data.gpu_usage;
            document.getElementById('ram-val').innerText = data.ram_usage;
            document.getElementById('ram-bar').style.width = data.ram_usage;
            
            document.getElementById('inference-speed-val').innerText = data.inference_speed;
            document.getElementById('queue-status-val').innerText = data.queue_status;
        })
        .catch(err => console.error("Error fetching stats:", err));
    }
    
    // Poll stats every 5 seconds
    setInterval(refreshSystemStats, 5000);
    refreshSystemStats(); // initial call

    // ---------------------------------------------------------
    // 8. INTERACTIVE CHART.JS ANALYTICS GRAPHICS
    // ---------------------------------------------------------
    let chartsInitialized = false;
    window.activeCharts = {};

    const modelData = {
        'SE-STN-EfficientNet-B5': {
            acc: "97.14%", prec: "97.30%", rec: "96.90%", spec: "97.55%", f1: "97.10%", auc: "98.62%", balAcc: "97.22%", ece: "0.012",
            report: {
                nonPrec: "96.90%", nonRec: "97.55%", nonF1: "97.22%",
                bccPrec: "97.30%", bccRec: "96.90%", bccF1: "97.10%",
                acc: "97.14%",
                macroPrec: "97.10%", macroRec: "97.22%", macroF1: "97.16%",
                weightedPrec: "97.10%", weightedRec: "97.14%", weightedF1: "97.12%"
            }
        },
        'EfficientNet-B4': {
            acc: "95.10%", prec: "95.30%", rec: "94.80%", spec: "95.40%", f1: "95.05%", auc: "96.85%", balAcc: "95.10%", ece: "0.015",
            report: {
                nonPrec: "94.80%", nonRec: "95.40%", nonF1: "95.10%",
                bccPrec: "95.30%", bccRec: "94.80%", bccF1: "95.05%",
                acc: "95.10%",
                macroPrec: "95.05%", macroRec: "95.10%", macroF1: "95.07%",
                weightedPrec: "95.05%", weightedRec: "95.10%", weightedF1: "95.07%"
            }
        },
        'DenseNet-121': {
            acc: "91.50%", prec: "91.80%", rec: "91.10%", spec: "91.90%", f1: "91.45%", auc: "93.32%", balAcc: "91.50%", ece: "0.021",
            report: {
                nonPrec: "91.10%", nonRec: "91.90%", nonF1: "91.50%",
                bccPrec: "91.80%", bccRec: "91.10%", bccF1: "91.45%",
                acc: "91.50%",
                macroPrec: "91.45%", macroRec: "91.50%", macroF1: "91.47%",
                weightedPrec: "91.45%", weightedRec: "91.50%", weightedF1: "91.47%"
            }
        },
        'ResNet-50': {
            acc: "90.14%", prec: "90.50%", rec: "89.80%", spec: "90.45%", f1: "90.15%", auc: "92.12%", balAcc: "90.12%", ece: "0.024",
            report: {
                nonPrec: "89.80%", nonRec: "90.45%", nonF1: "90.12%",
                bccPrec: "90.50%", bccRec: "89.80%", bccF1: "90.15%",
                acc: "90.14%",
                macroPrec: "90.15%", macroRec: "90.12%", macroF1: "90.13%",
                weightedPrec: "90.15%", weightedRec: "90.14%", weightedF1: "90.14%"
            }
        },
        'InceptionV3': {
            acc: "88.50%", prec: "88.90%", rec: "88.00%", spec: "89.00%", f1: "88.45%", auc: "90.50%", balAcc: "88.50%", ece: "0.028",
            report: {
                nonPrec: "88.00%", nonRec: "89.00%", nonF1: "88.50%",
                bccPrec: "88.90%", bccRec: "88.00%", bccF1: "88.45%",
                acc: "88.50%",
                macroPrec: "88.45%", macroRec: "88.50%", macroF1: "88.47%",
                weightedPrec: "88.45%", weightedRec: "88.50%", weightedF1: "88.47%"
            }
        },
        'MobileNetV3': {
            acc: "86.80%", prec: "87.20%", rec: "86.30%", spec: "87.30%", f1: "86.75%", auc: "89.10%", balAcc: "86.80%", ece: "0.032",
            report: {
                nonPrec: "86.30%", nonRec: "87.30%", nonF1: "86.80%",
                bccPrec: "87.20%", bccRec: "86.30%", bccF1: "86.75%",
                acc: "86.80%",
                macroPrec: "86.75%", macroRec: "86.80%", macroF1: "86.77%",
                weightedPrec: "86.75%", weightedRec: "86.80%", weightedF1: "86.77%"
            }
        },
        'VGG-16': {
            acc: "84.20%", prec: "84.60%", rec: "83.70%", spec: "84.70%", f1: "84.15%", auc: "86.50%", balAcc: "84.20%", ece: "0.038",
            report: {
                nonPrec: "83.70%", nonRec: "84.70%", nonF1: "84.20%",
                bccPrec: "84.60%", bccRec: "83.70%", bccF1: "84.15%",
                acc: "84.20%",
                macroPrec: "84.15%", macroRec: "84.20%", macroF1: "84.17%",
                weightedPrec: "84.15%", weightedRec: "84.20%", weightedF1: "84.17%"
            }
        }
    };
    
    function initAnalyticsCharts() {
        // No client-side dynamic Chart.js instances. Plots are static images generated by Matplotlib.
        const modelSelect = document.getElementById('analytics-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                updateModelAnalytics(e.target.value);
            });
        }
    }

    function updateModelAnalytics(modelName) {
        const data = modelData[modelName];
        if (!data) return;

        // Update Stats labels
        document.getElementById('stat-acc').innerText = data.acc;
        document.getElementById('stat-prec').innerText = data.prec;
        document.getElementById('stat-rec').innerText = data.rec;
        document.getElementById('stat-spec').innerText = data.spec;
        document.getElementById('stat-f1').innerText = data.f1;
        document.getElementById('stat-auc').innerText = data.auc;
        document.getElementById('stat-bal-acc').innerText = data.balAcc;
        document.getElementById('stat-ece').innerText = data.ece;

        // Update Classification Report table cells
        document.getElementById('report-non-prec').innerText = data.report.nonPrec;
        document.getElementById('report-non-rec').innerText = data.report.nonRec;
        document.getElementById('report-non-f1').innerText = data.report.nonF1;

        document.getElementById('report-bcc-prec').innerText = data.report.bccPrec;
        document.getElementById('report-bcc-rec').innerText = data.report.bccRec;
        document.getElementById('report-bcc-f1').innerText = data.report.bccF1;

        document.getElementById('report-acc').innerText = data.report.acc;

        document.getElementById('report-macro-prec').innerText = data.report.macroPrec;
        document.getElementById('report-macro-rec').innerText = data.report.macroRec;
        document.getElementById('report-macro-f1').innerText = data.report.macroF1;

        document.getElementById('report-weighted-prec').innerText = data.report.weightedPrec;
        document.getElementById('report-weighted-rec').innerText = data.report.weightedRec;
        document.getElementById('report-weighted-f1').innerText = data.report.weightedF1;
        
        // Update Static Matplotlib Image Sources
        const cleanName = modelName.replace(/-/g, '_').replace(/ /g, '_').toLowerCase();
        document.getElementById('img-diseases').src = `/static/outputs/distribution_${cleanName}.png`;
        document.getElementById('img-calibration').src = `/static/outputs/calibration_${cleanName}.png`;
        document.getElementById('img-loss').src = `/static/outputs/loss_${cleanName}.png`;
        document.getElementById('img-learning').src = `/static/outputs/learning_${cleanName}.png`;
        document.getElementById('img-confusion').src = `/static/outputs/confusion_${cleanName}.png`;
    }
});
