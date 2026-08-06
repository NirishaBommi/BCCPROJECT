/* ==========================================================================
   BCC Skin Lesion Detection Dashboard JavaScript Logic
   Handles: Tab switching, File Drag/Drop, AJAX file uploading, Dynamic result rendering
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Sidebar Tab Switching
    const navButtons = document.querySelectorAll('.nav-menu .nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Deactivate all nav buttons and tabs
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));

            // Activate target
            btn.classList.add('active');
            const activeTab = document.getElementById(targetTab);
            if (activeTab) {
                activeTab.classList.add('active');
            }
        });
    });

    // 2. Visual Comparison Sub-Tabs (Inference Panel)
    const visualButtons = document.querySelectorAll('.visuals-nav-btn');
    const visualPanels = document.querySelectorAll('.visual-panel');

    visualButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetVisual = btn.getAttribute('data-visual');

            visualButtons.forEach(b => b.classList.remove('active'));
            visualPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const activePanel = document.getElementById(targetVisual);
            if (activePanel) {
                activePanel.classList.add('active');
            }
        });
    });

    // 3. File Upload and Drag & Drop Handlers
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropZoneContent = dropZone.querySelector('.drop-zone-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');

    let selectedFile = null;

    // Trigger click on file input when clicking drop zone
    dropZone.addEventListener('click', (e) => {
        // Prevent click if we clicked inside preview container or if file already uploaded
        if (e.target.closest('#preview-container') || selectedFile) return;
        fileInput.click();
    });

    // Drag-over effects
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped file
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle selected file
    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Error: Please upload a valid image file (JPG or PNG).');
            return;
        }

        selectedFile = file;
        
        // Show file preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.style.display = 'none';
            previewContainer.style.display = 'block';
            
            // Toggle buttons
            analyzeBtn.disabled = false;
            resetBtn.style.display = 'inline-flex';
        };
        reader.readAsDataURL(file);
    }

    // Reset handler
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropZoneContent.style.display = 'flex';
        
        analyzeBtn.disabled = true;
        resetBtn.style.display = 'none';
        
        // Reset results display
        document.getElementById('placeholder-results').style.display = 'block';
        document.getElementById('loading-results').style.display = 'none';
        document.getElementById('actual-results').style.display = 'none';
        document.getElementById('threshold-tick').style.display = 'none';
        document.getElementById('threshold-explanation').style.display = 'none';
    });

    // 4. AJAX Form Upload & Model Inference Execution
    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Add settings parameters
        const modelType = document.getElementById('model-type-select').value;
        const useTta = document.getElementById('tta-toggle').checked;
        const useThreshold = document.getElementById('threshold-toggle').checked;
        
        formData.append('model_type', modelType);
        formData.append('use_tta', useTta);
        formData.append('use_threshold', useThreshold);

        // UI State: Loading
        document.getElementById('placeholder-results').style.display = 'none';
        document.getElementById('loading-results').style.display = 'flex';
        document.getElementById('actual-results').style.display = 'none';
        
        analyzeBtn.disabled = true;
        resetBtn.style.display = 'none';

        // Call backend predict route
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error') });
            }
            return response.json();
        })
        .then(data => {
            // UI State: Success
            document.getElementById('loading-results').style.display = 'none';
            const actualResults = document.getElementById('actual-results');
            actualResults.style.display = 'block';
            
            analyzeBtn.disabled = false;
            resetBtn.style.display = 'inline-flex';

            // Populate text metrics
            const classNameEl = document.getElementById('result-class-name');
            classNameEl.innerText = data.predicted_label;
            
            const badgeEl = document.getElementById('result-badge');
            
            // We want the gauge to represent the BCC Risk Probability (probability of the cancer class)
            const bccPct = (data.probability_bcc * 100).toFixed(2);
            document.getElementById('result-confidence').innerText = `${bccPct}%`;
            
            // Adjust risk badge styling
            if (data.predicted_class === 1) { // BCC
                classNameEl.style.color = 'var(--danger)';
                badgeEl.innerText = 'High Risk';
                badgeEl.className = 'badge badge-danger';
            } else { // Non-BCC
                classNameEl.style.color = 'var(--success)';
                badgeEl.innerText = 'Low Risk';
                badgeEl.className = 'badge badge-success';
            }

            // Animate confidence meter (based on BCC risk probability)
            const fillBar = document.getElementById('result-confidence-fill');
            fillBar.style.width = '0%';
            setTimeout(() => {
                fillBar.style.width = `${bccPct}%`;
                if (data.predicted_class === 1) {
                    fillBar.style.background = 'linear-gradient(90deg, #f093fb, #ef4444)';
                } else {
                    fillBar.style.background = 'linear-gradient(90deg, #4facfe, #10b981)';
                }
            }, 100);

            // Update threshold tick and text explanation dynamically
            const thresholdTick = document.getElementById('threshold-tick');
            const explanationEl = document.getElementById('threshold-explanation');
            const threshUsed = data.threshold_used || 0.40;
            const bccProb = data.probability_bcc;
            
            thresholdTick.style.display = 'block';
            thresholdTick.style.left = `${threshUsed * 100}%`;
            thresholdTick.style.setProperty('--threshold-label', `"T=${(threshUsed * 100).toFixed(0)}%"`);
            
            if (data.predicted_class === 1) { // High Risk (BCC)
                if (bccProb < 0.50) {
                    explanationEl.style.display = 'flex';
                    explanationEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span><strong>Screening Flag:</strong> Risk probability (${(bccProb * 100).toFixed(1)}%) is under 50% but exceeds the clinical screening threshold of ${(threshUsed * 100).toFixed(0)}% to prevent false negatives.</span>`;
                } else {
                    explanationEl.style.display = 'flex';
                    explanationEl.innerHTML = `<i class="fa-solid fa-shield-halved"></i> <span><strong>High Risk:</strong> Risk probability (${(bccProb * 100).toFixed(1)}%) is above the ${(threshUsed * 100).toFixed(0)}% threshold. Professional dermatological evaluation is recommended.</span>`;
                }
            } else { // Low Risk (Non-BCC)
                explanationEl.style.display = 'flex';
                explanationEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span><strong>Low Risk:</strong> Risk probability (${(bccProb * 100).toFixed(1)}%) is safely below the active screening threshold of ${(threshUsed * 100).toFixed(0)}%.</span>`;
            }

            // Detailed probabilities
            document.getElementById('prob-bcc-val').innerText = `${(data.probability_bcc * 100).toFixed(2)}%`;
            document.getElementById('prob-nonbcc-val').innerText = `${(data.probability_non_bcc * 100).toFixed(2)}%`;

            // Populate images
            document.getElementById('cam-orig-img').src = data.orig_url;
            document.getElementById('cam-overlay-img').src = data.cam_url;
            document.getElementById('prep-orig-img').src = data.orig_url;
            document.getElementById('prep-normalized-img').src = data.prep_url;
        })
        .catch(error => {
            // UI State: Error
            console.error('Error analyzing image:', error);
            alert(`Analysis failed: ${error.message}`);
            
            document.getElementById('placeholder-results').style.display = 'block';
            document.getElementById('loading-results').style.display = 'none';
            
            analyzeBtn.disabled = false;
            resetBtn.style.display = 'inline-flex';
        });
    });
});
