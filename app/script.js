// document.addEventListener("DOMContentLoaded", () => {
    
//     // --- Get DOM Elements ---
//     const loadImageBtn = document.getElementById("load-image-btn");
//     const imageIdInput = document.getElementById("image-id");
//     const maskToggle = document.getElementById("show-mask-toggle");
//     const uploadInput = document.getElementById("upload-image-input");
//     const fileDisplay = document.getElementById("file-name-display"); // <-- NEW
    
//     const imageDisplay = document.getElementById("image-display");
//     const resultMessage = document.getElementById("result-message");
//     const debugInfo = document.getElementById("debug-info");

//     // --- State ---
//     let currentImageId = null;
//     let currentImageData = null; 
//     let uploadedFile = null; 

//     // --- Helper Functions (Same) ---
//     function setResultMessage(status, message) {
//         resultMessage.textContent = message;
//         resultMessage.className = 'result-' + status; 
//     }

//     function formatPercent(value) {
//         return (value * 100).toFixed(2) + '%';
//     }

//     // --- Core Functions ---
//     async function loadImagePlot() {
//         // --- MODIFY THIS FUNCTION ---
//         // 1. Clear "Upload Mode"
//         uploadedFile = null;
//         uploadInput.value = null; 
//         fileDisplay.textContent = "No new plot added"; // <-- ADD THIS LINE
        
//         // 2. Set "ID Mode" state
//         currentImageId = imageIdInput.value;
//         setResultMessage('info', `Loading image ${currentImageId}...`);
//         debugInfo.innerHTML = "<p>N/A</p>"; 

//         try {
//             // (Rest of the function is the same)
//             const response = await fetch(`/api/image_data/${currentImageId}`);
//             if (!response.ok) {
//                 const errorData = await response.json();
//                 throw new Error(errorData.detail || `Image ${currentImageId} not found.`);
//             }
            
//             currentImageData = await response.json();
//             displayImage(); 
//             setResultMessage('info', 'Image loaded. Click a sub-plot (1-9) to check.');
            
//         } catch (error) {
//             setResultMessage('denied', error.message);
//             imageDisplay.innerHTML = `<p class="placeholder-text">Error loading image.</p>`;
//         }
//     }

//     function loadUploadedPlot(event) {
//         // --- MODIFY THIS FUNCTION ---
//         const file = event.target.files[0];
//         if (!file) {
//             fileDisplay.textContent = "No new plot added"; // <-- ADD THIS
//             return;
//         }

//         // 1. Set "Upload Mode" state
//         uploadedFile = file;
//         currentImageId = null; 
//         currentImageData = null;
//         imageIdInput.value = ""; 
        
//         fileDisplay.textContent = file.name; // <-- SETS FILE NAME
        
//         // 2. Disable mask toggle
//         maskToggle.checked = false; 
//         maskToggle.disabled = true; 
        
//         setResultMessage('info', 'Image loaded. Click a sub-plot (1-9) to check.');
//         debugInfo.innerHTML = "<p>N/A</p>";
        
//         // 3. Use FileReader to display the image preview
//         const reader = new FileReader();
//         reader.onload = (e) => {
//             imageDisplay.innerHTML = `
//                 <img src="${e.target.result}" alt="Uploaded Image">
//                 <div class="grid-overlay">
//                     ${[...Array(9)].map((_, i) => `
//                         <div class="grid-cell" data-sub-plot-id="${i + 1}">${i + 1}</div>
//                     `).join('')}
//                 </div>
//             `;
//             // 4. Re-add listeners
//             document.querySelectorAll('.grid-cell').forEach(cell => {
//                 cell.addEventListener('click', onGridCellClick);
//             });
//         };
//         reader.readAsDataURL(file);
//     }

//     // --- (displayImage function is the same) ---
//     function displayImage() {
//         if (!currentImageData) return;
//         maskToggle.disabled = false;
//         const showMask = maskToggle.checked;
//         const imageUrl = showMask ? currentImageData.mask_url : currentImageData.image_url;
//         const altText = showMask ? "Ground Truth Mask" : "Original Image";
//         imageDisplay.innerHTML = `
//             <img src="${imageUrl}" alt="${altText} ${currentImageData.image_id}">
//             <div class="grid-overlay">
//                 ${[...Array(9)].map((_, i) => `
//                     <div class="grid-cell" data-sub-plot-id="${i + 1}">${i + 1}</div>
//                 `).join('')}
//             </div>
//         `;
//         document.querySelectorAll('.grid-cell').forEach(cell => {
//             cell.addEventListener('click', onGridCellClick);
//         });
//     }

//     // --- (onGridCellClick function is the same) ---
//     async function onGridCellClick(event) {
//         const subPlotId = event.target.dataset.subPlotId;
//         if (!subPlotId) return; 

//         setResultMessage('info', `Running AI analysis on Sub-plot ${subPlotId}...`);
//         debugInfo.innerHTML = "<p>Loading...</p>";
        
//         try {
//             let response;
            
//             if (uploadedFile) {
//                 // --- UPLOAD MODE ---
//                 const formData = new FormData();
//                 formData.append("file", uploadedFile);
//                 response = await fetch(`/api/analyze_upload/${subPlotId}`, {
//                     method: 'POST',
//                     body: formData
//                 });
                
//             } else if (currentImageId) {
//                 // --- ID MODE ---
//                 response = await fetch(`/api/check_plot/${currentImageId}/${subPlotId}`);
                
//             } else {
//                 throw new Error("Please load an image first.");
//             }

//             // 4. Handle the response (same for both modes)
//             const result = await response.json();
//             if (!response.ok) {
//                 throw new Error(result.detail || 'Error checking plot.');
//             }

//             setResultMessage(result.status, result.message);
            
//             if (result.data) {
//                 let html = "";
//                 for (const [key, value] of Object.entries(result.data)) {
//                     let displayValue = value;
//                     if (key.includes("Ratio")) {
//                         displayValue = formatPercent(value);
//                     }
//                     html += `<p><strong>${key}:</strong> ${displayValue}</p>`;
//                 }
//                 debugInfo.innerHTML = html;
//             }

//         } catch (error) {
//             setResultMessage('denied', error.message);
//             debugInfo.innerHTML = `<p style="color: red;">${error.message}</p>`;
//         }
//     }

//     // --- (Event Listeners are the same) ---
//     loadImageBtn.addEventListener("click", loadImagePlot);
//     imageIdInput.addEventListener("keyup", (event) => {
//         if (event.key === "Enter") {
//             loadImagePlot();
//         }
//     });
//     uploadInput.addEventListener("change", loadUploadedPlot);
//     maskToggle.addEventListener("change", displayImage);
// });

/**
 * =========================================
 * ECOPLANNER AI - FRONTEND CONTROLLER
 * =========================================
 * Handles user interactions, state management,
 * and API communication.
 */

document.addEventListener("DOMContentLoaded", () => {

    // --- DOM ELEMENTS REFERENCE ---
    const refs = {
        loadImageBtn: document.getElementById("load-image-btn"),
        imageIdInput: document.getElementById("image-id"),
        maskToggle: document.getElementById("show-mask-toggle"),
        uploadInput: document.getElementById("upload-image-input"),
        fileNameDisplay: document.getElementById("file-name-display"),
        imageDisplay: document.getElementById("image-display"),
        resultBanner: document.getElementById("result-message"),
        debugInfo: document.getElementById("debug-info"),
        initialEmptyState: document.getElementById("initial-empty-state"),
        metricsEmptyState: document.getElementById("metrics-empty-state"),
    };

    // --- APPLICATION STATE ---
    const state = {
        currentImageId: null,
        currentImageData: null, // Stores {image_url, mask_url} from API
        uploadedFile: null,
        isLoading: false
    };

    // =========================================
    // VIEW UPDATE FUNCTIONS
    // =========================================

    /**
     * Updates the main status banner with a message and style.
     * @param {'neutral'|'success'|'danger'|'info'|'loading'} type
     * @param {string} message
     */
    function updateStatus(type, message) {
        refs.resultBanner.textContent = message;
        refs.resultBanner.className = `status-banner status-banner--${type}`;
        
        // Optional: Add a subtle pulse animation for loading state
        if (type === 'loading') {
             refs.resultBanner.innerHTML = `
                <svg class="animate-spin icon-sm" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                ${message}
             `;
        }
    }

    /**
     * Renders the image container with the 3x3 interactive grid.
     * @param {string} imageUrl - Source URL for the image
     * @param {string} altText - Accessibility text
     */
    function renderImageViewer(imageUrl, altText) {
        refs.imageDisplay.innerHTML = `
            <div class="image-container">
                <img src="${imageUrl}" alt="${altText}" loading="lazy">
                <div class="grid-overlay" role="grid" aria-label="Selectable image sectors">
                    ${Array.from({ length: 9 }, (_, i) => 
                        `<div class="grid-cell" role="gridcell" tabindex="0" 
                              aria-label="Sector ${i + 1}" data-sub-plot-id="${i + 1}">
                            ${i + 1}
                        </div>`
                    ).join('')}
                </div>
            </div>
        `;

        // Re-attach event listeners to the new grid cells
        refs.imageDisplay.querySelectorAll('.grid-cell').forEach(cell => {
            cell.addEventListener('click', handleGridCellClick);
            // Add keyboard support (Enter/Space to select)
            cell.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    cell.click();
                }
            });
        });
    }

    /**
     * Formats raw data into a readable metrics list.
     * @param {object} data - Key-value pairs of debug data
     */
    function renderMetrics(data) {
        if (!data) return;

        const metricsHtml = Object.entries(data).map(([key, value]) => {
            let displayValue = value;

            // Format percentages
            if (key.toLowerCase().includes("ratio")) {
                displayValue = (value * 100).toFixed(1) + '%';
            } 
            // Format large numbers with commas
            else if (typeof value === 'number' && value > 999) {
                displayValue = value.toLocaleString();
            }

            return `
                <div class="metric-item">
                    <span class="metric-label">${key}</span>
                    <span class="metric-value font-bold">${displayValue}</span>
                </div>
            `;
        }).join('');

        refs.debugInfo.innerHTML = `<div class="metrics-list fade-in">${metricsHtml}</div>`;
    }

    // =========================================
    // EVENT HANDLERS & API CALLS
    // =========================================

    async function handleLoadDbImage() {
        const id = refs.imageIdInput.value.trim();
        if (!id) {
             //Shake input if empty
             refs.imageIdInput.style.borderColor = 'var(--color-danger)';
             setTimeout(() => refs.imageIdInput.style.borderColor = '', 500);
             return;
        }

        // Reset UI State
        state.uploadedFile = null;
        refs.uploadInput.value = '';
        refs.fileNameDisplay.textContent = "No file selected";
        state.currentImageId = id;

        updateStatus('loading', `Retrieving Record #${id}...`);
        refs.debugInfo.innerHTML = '<p class="text-muted text-center">Loading...</p>';

        try {
            const response = await fetch(`/api/image_data/${id}`);
            if (!response.ok) throw new Error(`Database record ${id} not found.`);

            state.currentImageData = await response.json();
            
            // Update View
            updateStatus('neutral', 'Ready. Select a grid sector to analyze.');
            refs.metricsEmptyState ? refs.metricsEmptyState.style.display = 'block' : refs.debugInfo.innerHTML = '<p class="text-muted text-center">Waiting for selection...</p>';
            
            // Enable mask toggle for DB images
            refs.maskToggle.disabled = false;
            refs.maskToggle.checked = false;
            
            renderImageViewer(state.currentImageData.image_url, `Satellite Plot ${id}`);

        } catch (error) {
            console.error(error);
            updateStatus('danger', 'Error: ' + error.message);
            refs.imageDisplay.innerHTML = `
                <div class="empty-state text-danger">
                    <svg class="empty-state__icon" width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                    <h3>Failed to Load</h3>
                    <p>${error.message}</p>
                </div>`;
        }
    }

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Update State
        state.uploadedFile = file;
        state.currentImageId = null;
        state.currentImageData = null;

        // Update UI
        refs.imageIdInput.value = '';
        refs.fileNameDisplay.textContent = file.name;
        refs.maskToggle.disabled = true; // Disable masks for uploads
        refs.maskToggle.checked = false;

        updateStatus('neutral', 'File loaded. Select a sector to analyze.');

        // Render preview
        const reader = new FileReader();
        reader.onload = (e) => renderImageViewer(e.target.result, "Uploaded Satellite Image");
        reader.readAsDataURL(file);
    }

    async function handleGridCellClick(event) {
        const cell = event.target;
        const subPlotId = cell.dataset.subPlotId;
        if (!subPlotId || state.isLoading) return;

        // UI Feedback: Highlight selected cell
        document.querySelectorAll('.grid-cell').forEach(c => c.classList.remove('is-selected'));
        cell.classList.add('is-selected');

        state.isLoading = true;
        updateStatus('loading', `Analyzing Sector ${subPlotId}...`);
        
        // Show skeleton loader in metrics panel
        refs.debugInfo.innerHTML = `
            <div class="animate-pulse space-y-4">
                <div class="h-4 bg-slate-700 rounded w-3/4"></div>
                <div class="h-4 bg-slate-700 rounded"></div>
                <div class="h-4 bg-slate-700 rounded"></div>
                <div class="h-4 bg-slate-700 rounded w-5/6"></div>
            </div>`;

        try {
            let response;
            if (state.uploadedFile) {
                const formData = new FormData();
                formData.append("file", state.uploadedFile);
                response = await fetch(`/api/analyze_upload/${subPlotId}`, { method: 'POST', body: formData });
            } else if (state.currentImageId) {
                response = await fetch(`/api/check_plot/${state.currentImageId}/${subPlotId}`);
            } else {
                throw new Error("No active image. Please load a plot first.");
            }

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed.');
            }

            const result = await response.json();
            
            // Determine success/fail state based on 'status'
            const statusType = result.status === 'approved' ? 'success' : 'danger';
            updateStatus(statusType, result.message);
            renderMetrics(result.data);

        } catch (error) {
            console.error(error);
            updateStatus('danger', error.message);
            refs.debugInfo.innerHTML = `<p class="text-danger">Analysis failed. Please try again.</p>`;
        } finally {
            state.isLoading = false;
        }
    }

    function toggleMaskDisplay() {
        if (!state.currentImageData) return;
        const showMask = refs.maskToggle.checked;
        const url = showMask ? state.currentImageData.mask_url : state.currentImageData.image_url;
        const alt = showMask ? "Ground Truth Segmentation Mask" : `Satellite Plot ${state.currentImageId}`;
        renderImageViewer(url, alt);
    }

    // =========================================
    // INITIALIZATION
    // =========================================
    
    // Attach Event Listeners
    refs.loadImageBtn.addEventListener('click', handleLoadDbImage);
    refs.imageIdInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleLoadDbImage(); });
    refs.uploadInput.addEventListener('change', handleFileUpload);
    refs.maskToggle.addEventListener('change', toggleMaskDisplay);

});