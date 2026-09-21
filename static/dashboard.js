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
        'EfficientNet-B5': {
            roc: [0.0, 0.94, 0.98, 0.99, 1.0, 1.0],
            pr: [1.0, 0.99, 0.98, 0.97, 0.92, 0.23],
            lossTrain: [0.45, 0.28, 0.18, 0.12, 0.09, 0.08],
            lossVal: [0.48, 0.32, 0.22, 0.15, 0.11, 0.09],
            cal: [18, 38, 62, 79, 97],
            diseases: [42, 15, 30, 8, 5]
        },
        'EfficientNet-B0': {
            roc: [0.0, 0.88, 0.94, 0.96, 0.98, 1.0],
            pr: [1.0, 0.96, 0.92, 0.88, 0.82, 0.18],
            lossTrain: [0.55, 0.38, 0.26, 0.18, 0.14, 0.12],
            lossVal: [0.59, 0.42, 0.31, 0.22, 0.18, 0.14],
            cal: [16, 36, 58, 76, 94],
            diseases: [38, 18, 28, 10, 6]
        },
        'ResNet50': {
            roc: [0.0, 0.84, 0.91, 0.93, 0.96, 1.0],
            pr: [1.0, 0.93, 0.88, 0.84, 0.76, 0.15],
            lossTrain: [0.62, 0.45, 0.32, 0.24, 0.19, 0.16],
            lossVal: [0.66, 0.49, 0.38, 0.28, 0.23, 0.18],
            cal: [14, 34, 54, 72, 90],
            diseases: [35, 20, 26, 12, 7]
        },
        'DenseNet121': {
            roc: [0.0, 0.86, 0.92, 0.94, 0.97, 1.0],
            pr: [1.0, 0.94, 0.90, 0.85, 0.79, 0.16],
            lossTrain: [0.58, 0.41, 0.29, 0.21, 0.16, 0.14],
            lossVal: [0.62, 0.45, 0.34, 0.25, 0.20, 0.16],
            cal: [15, 35, 56, 74, 92],
            diseases: [36, 19, 27, 11, 7]
        }
    };
    
    function initAnalyticsCharts() {
        if (chartsInitialized) return;
        chartsInitialized = true;
        
        // 1. Learning Curve (Accuracy) - Image 1
        const ctxAccuracy = document.getElementById('chart-accuracy')?.getContext('2d');
        if (ctxAccuracy) {
            window.activeCharts.accuracy = new Chart(ctxAccuracy, {
                type: 'line',
                data: {
                    labels: ['5', '10', '15', '20', '25', '30'],
                    datasets: [
                        {
                            label: 'Training Accuracy',
                            data: [72, 81, 88, 92.5, 95, 97],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Validation Accuracy',
                            data: [70, 79, 86, 91, 94, 96.8],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Training Epochs', color: '#94a3b8' } },
                        y: { title: { display: true, text: 'Model Accuracy (%)', color: '#94a3b8' }, min: 65, max: 100 }
                    }
                }
            });
        }
        
        // 2. Loss Convergence (FL-CE) - Image 2
        const ctxLoss = document.getElementById('chart-loss')?.getContext('2d');
        if (ctxLoss) {
            window.activeCharts.loss = new Chart(ctxLoss, {
                type: 'line',
                data: {
                    labels: ['5', '10', '15', '20', '25', '30'],
                    datasets: [
                        {
                            label: 'Training Loss',
                            data: [0.58, 0.42, 0.28, 0.18, 0.11, 0.06],
                            borderColor: '#ef4444',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Validation Loss',
                            data: [0.42, 0.31, 0.20, 0.12, 0.08, 0.05],
                            borderColor: '#3b82f6',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Training Epochs', color: '#94a3b8' } },
                        y: { title: { display: true, text: 'Cross-Entropy Loss (FL-CE)', color: '#94a3b8' }, min: 0.0, max: 0.65 }
                    }
                }
            });
        }
        
        // 3. Class Distribution Pie Chart - Image 4
        const ctxDiseases = document.getElementById('chart-diseases')?.getContext('2d');
        if (ctxDiseases) {
            window.activeCharts.diseases = new Chart(ctxDiseases, {
                type: 'doughnut',
                data: {
                    labels: ['BCC (42%)', 'Melanoma (15%)', 'Benign Nevus (30%)', 'Keratosis (8%)', 'Other (5%)'],
                    datasets: [{
                        data: [42.0, 15.0, 30.0, 8.0, 5.0],
                        backgroundColor: ['#dc2626', '#ea580c', '#16a34a', '#9333ea', '#6b7280']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }
        
        // 4. Comparative ROC Curves - Image 5
        const ctxRoc = document.getElementById('chart-roc')?.getContext('2d');
        if (ctxRoc) {
            window.activeCharts.roc = new Chart(ctxRoc, {
                type: 'line',
                data: {
                    labels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    datasets: [
                        { label: 'SE-STN-EfficientNet-B5 (AUC = 0.978)', data: [0.0, 0.94, 0.98, 0.99, 1.0, 1.0], borderColor: '#0284c7', borderWidth: 3, fill: false },
                        { label: 'EfficientNet-B4 (AUC = 0.961)', data: [0.0, 0.91, 0.96, 0.98, 0.99, 1.0], borderColor: '#38bdf8', borderWidth: 2, borderDash: [4, 4], fill: false },
                        { label: 'DenseNet-121 (AUC = 0.952)', data: [0.0, 0.86, 0.92, 0.94, 0.97, 1.0], borderColor: '#ea580c', borderWidth: 2, borderDash: [6, 6], fill: false },
                        { label: 'ResNet-50 (AUC = 0.941)', data: [0.0, 0.84, 0.90, 0.93, 0.96, 1.0], borderColor: '#f97316', borderWidth: 2, borderDash: [2, 2], fill: false },
                        { label: 'InceptionV3 (AUC = 0.937)', data: [0.0, 0.81, 0.88, 0.91, 0.94, 1.0], borderColor: '#16a34a', borderWidth: 2, borderDash: [3, 3], fill: false },
                        { label: 'MobileNetV3 (AUC = 0.930)', data: [0.0, 0.78, 0.85, 0.89, 0.93, 1.0], borderColor: '#4ade80', borderWidth: 2, borderDash: [1, 1], fill: false },
                        { label: 'VGG-16 (AUC = 0.923)', data: [0.0, 0.74, 0.81, 0.85, 0.90, 1.0], borderColor: '#dc2626', borderWidth: 2, borderDash: [4, 2], fill: false },
                        { label: 'Random (AUC = 0.5000)', data: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], borderColor: '#6b7280', borderWidth: 1.5, borderDash: [8, 4], fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' } },
                        y: { title: { display: true, text: 'True Positive Rate (Sensitivity)', color: '#94a3b8' }, min: 0.0, max: 1.0 }
                    }
                }
            });
        }
        
        // 5. Comparative Precision-Recall Curves - Image 6 (Section 5.5)
        const ctxPr = document.getElementById('chart-pr')?.getContext('2d');
        if (ctxPr) {
            window.activeCharts.pr = new Chart(ctxPr, {
                type: 'line',
                data: {
                    labels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    datasets: [
                        { label: 'SE-STN-EfficientNet-B5 (AP = 0.968)', data: [1.0, 0.99, 0.98, 0.97, 0.92, 0.23], borderColor: '#0284c7', borderWidth: 3, fill: false },
                        { label: 'EfficientNet-B4 (AP = 0.938)', data: [1.0, 0.97, 0.95, 0.93, 0.86, 0.20], borderColor: '#38bdf8', borderWidth: 2, borderDash: [4, 4], fill: false },
                        { label: 'DenseNet-121 (AP = 0.895)', data: [1.0, 0.94, 0.91, 0.87, 0.79, 0.16], borderColor: '#ea580c', borderWidth: 2, borderDash: [6, 6], fill: false },
                        { label: 'ResNet-50 (AP = 0.884)', data: [1.0, 0.93, 0.89, 0.85, 0.76, 0.15], borderColor: '#f97316', borderWidth: 2, borderDash: [2, 2], fill: false },
                        { label: 'InceptionV3 (AP = 0.855)', data: [1.0, 0.90, 0.85, 0.81, 0.72, 0.14], borderColor: '#16a34a', borderWidth: 2, borderDash: [3, 3], fill: false },
                        { label: 'MobileNetV3 (AP = 0.832)', data: [1.0, 0.88, 0.82, 0.78, 0.68, 0.12], borderColor: '#4ade80', borderWidth: 2, borderDash: [1, 1], fill: false },
                        { label: 'VGG-16 (AP = 0.798)', data: [1.0, 0.86, 0.79, 0.74, 0.63, 0.10], borderColor: '#dc2626', borderWidth: 2, borderDash: [4, 2], fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Recall (Sensitivity)', color: '#94a3b8' } },
                        y: { title: { display: true, text: 'Precision (Positive Predictive Value)', color: '#94a3b8' }, min: 0.0, max: 1.05 }
                    }
                }
            });
        }
        
        // 6. Reliability Calibration Diagram - Image 7 (Section 5.6)
        const ctxCal = document.getElementById('chart-calibration')?.getContext('2d');
        if (ctxCal) {
            window.activeCharts.calibration = new Chart(ctxCal, {
                type: 'bar',
                data: {
                    labels: ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5'],
                    datasets: [
                        { label: 'Perfect Calibration', data: [20.0, 40.0, 60.0, 80.0, 100.0], type: 'line', borderColor: '#94a3b8', borderWidth: 2, borderDash: [5, 5], fill: false, pointBackgroundColor: '#64748b' },
                        { label: 'Model Confidence', data: [18.0, 38.0, 62.0, 79.0, 97.0], backgroundColor: '#0284c7', borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Mean Predicted Probability (Bins)', color: '#94a3b8' } },
                        y: { title: { display: true, text: 'Fraction of Positives (%)', color: '#94a3b8' }, min: 0, max: 105 }
                    }
                }
            });
        }

        // Add model profile change listener
        const modelSelect = document.getElementById('analytics-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                updateModelAnalytics(e.target.value);
            });
        }
    }

    function updateModelAnalytics(modelName) {
        const data = modelData[modelName];
        if (!data || !window.activeCharts) return;
        
        // Update ROC dataset
        if (window.activeCharts.roc) {
            window.activeCharts.roc.data.datasets[0].data = data.roc;
            const aucVal = modelName === 'EfficientNet-B5' ? '0.986' : modelName === 'EfficientNet-B0' ? '0.941' : modelName === 'ResNet50' ? '0.918' : '0.924';
            window.activeCharts.roc.data.datasets[0].label = `ROC Curve (AUC = ${aucVal})`;
            window.activeCharts.roc.update();
        }
        
        // Update PR dataset
        if (window.activeCharts.pr) {
            window.activeCharts.pr.data.datasets[0].data = data.pr;
            const prAuc = modelName === 'EfficientNet-B5' ? '0.968' : modelName === 'EfficientNet-B0' ? '0.912' : modelName === 'ResNet50' ? '0.884' : '0.895';
            window.activeCharts.pr.data.datasets[0].label = `PR Curve (AUC = ${prAuc})`;
            window.activeCharts.pr.update();
        }
        
        // Update Loss curves
        if (window.activeCharts.loss) {
            window.activeCharts.loss.data.datasets[0].data = data.lossTrain;
            window.activeCharts.loss.data.datasets[1].data = data.lossVal;
            window.activeCharts.loss.update();
        }
        
        // Update Calibration bins
        if (window.activeCharts.calibration) {
            window.activeCharts.calibration.data.datasets[1].data = data.cal;
            window.activeCharts.calibration.update();
        }
        
        // Update disease doughnut slice
        if (window.activeCharts.diseases) {
            window.activeCharts.diseases.data.datasets[0].data = data.diseases;
            window.activeCharts.diseases.update();
        }
    }
});
