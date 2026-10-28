// Initialize Chart.js
let sortingChart = null;
let ctx = null;

// Color mapping for ArUco markers
const colorMap = {
    'Red': 'rgb(255, 0, 0)',
    'Blue': 'rgb(0, 0, 255)',
    'Green': 'rgb(0, 255, 0)',
    'Yellow': 'rgb(255, 255, 0)',
    'Orange': 'rgb(255, 165, 0)',
    'Black': 'rgb(0, 0, 0)',
    'White': 'rgb(255, 255, 255)',
    'Purple': 'rgb(128, 0, 128)'
};

// Initialize chart
function initChart() {
    const canvas = document.getElementById('sortingChart');
    if (!canvas) {
        console.error('Chart canvas not found');
        return;
    }
    
    ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Could not get 2D context');
        return;
    }
    
    // Ensure canvas has proper dimensions
    const container = canvas.parentElement;
    if (container) {
        canvas.width = container.clientWidth - 30; // Account for padding
        canvas.height = container.clientHeight - 30;
    }
    
    sortingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'LEGOs Sorted by Color',
                    color: '#ffffff',
                    font: {
                        size: 14
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#ffffff',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            const count = context.parsed.y;
                            const color = context.label;
                            // Get aruco_id from dataset metadata
                            const dataset = context.dataset;
                            const arucoId = (dataset.aruco_ids && dataset.aruco_ids[color]) || 'N/A';
                            return [
                                `Count: ${count}`,
                                `ArUco Marker ID: ${arucoId}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#ffffff',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
            y: {
                ticks: {
                    color: '#ffffff',
                    font: {
                        size: 12
                    },
                    stepSize: 1,
                    precision: 0
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                beginAtZero: true,
                min: 0,
                title: {
                    display: true,
                    text: 'Total Count',
                    color: '#ffffff',
                    font: {
                        size: 12
                    }
                }
            }
            }
        }
    });
}

// Update database table (fetches fresh data from PostgreSQL)
function updateDatabase() {
    // Add cache-busting to ensure fresh data from database
    fetch(`/api/database?t=${Date.now()}`, {
        cache: 'no-cache',
        headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            
            if (data && data.length > 0) {
                data.forEach(record => {
                    const row = document.createElement('tr');
                    const statusClass = `status-${record.status.toLowerCase()}`;
                    
                    row.innerHTML = `
                        <td>${record.id}</td>
                        <td>${record.timestamp.split(' ')[1]}</td>
                        <td style="color: ${colorMap[record.color] || '#ffffff'}">${record.color}</td>
                        <td>${record.aruco_marker_id}</td>
                        <td>${record.count}</td>
                        <td class="${statusClass}">${record.status}</td>
                    `;
                    tbody.appendChild(row);
                });
            } else {
                // Show message if no data
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No records found</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching database:', error);
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #ff0000;">Error loading data</td></tr>';
        });
}

// Update statistics (fetches fresh data from PostgreSQL)
function updateStatistics() {
    fetch(`/api/statistics?t=${Date.now()}`, {
        cache: 'no-cache',
        headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('stats').textContent = 
                `Total Sorted: ${data.total} | Processing: ${data.processing} | Completed: ${data.completed}`;
        })
        .catch(error => {
            console.error('Error fetching statistics:', error);
            document.getElementById('stats').textContent = 'Error loading statistics';
        });
}

// Update graph (fetches fresh data from PostgreSQL)
function updateGraph() {
    fetch(`/api/graph_data?t=${Date.now()}`, {
        cache: 'no-cache',
        headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (!sortingChart) {
                console.error('Chart not initialized');
                return;
            }
            
            if (data && data.labels && data.datasets && data.datasets.length > 0) {
                const dataset = data.datasets[0];
                const labels = data.labels || [];
                
                // Ensure data is array of numbers
                let processedData = [];
                if (Array.isArray(dataset.data)) {
                    processedData = dataset.data.map((count, index) => {
                        const numValue = typeof count === 'number' ? count : Number(count || 0);
                        return Math.max(0, numValue); // Ensure non-negative
                    });
                }
                
                console.log('Updating chart - Labels:', labels, 'Data:', processedData);
                
                // Ensure arrays match in length
                if (processedData.length !== labels.length) {
                    // Pad or trim to match labels
                    if (processedData.length < labels.length) {
                        while (processedData.length < labels.length) {
                            processedData.push(0);
                        }
                    } else {
                        processedData = processedData.slice(0, labels.length);
                    }
                }
                
                // Get colors - ensure we have colors for all labels
                const backgroundColors = Array.isArray(dataset.backgroundColor) ? [...dataset.backgroundColor] : [];
                const borderColors = Array.isArray(dataset.borderColor) ? [...dataset.borderColor] : [];
                
                // Pad colors if needed
                while (backgroundColors.length < labels.length) {
                    backgroundColors.push('rgb(128, 128, 128)');
                }
                while (borderColors.length < labels.length) {
                    borderColors.push('rgb(255, 255, 255)');
                }
                
                // Update chart data directly
                sortingChart.data.labels = labels;
                sortingChart.data.datasets = [{
                    label: dataset.label || 'Total Count',
                    data: processedData,
                    backgroundColor: backgroundColors.slice(0, labels.length),
                    borderColor: borderColors.slice(0, labels.length),
                    borderWidth: dataset.borderWidth || 2,
                    aruco_ids: data.aruco_ids || {}
                }];
                
                // Resize chart if needed
                const canvas = document.getElementById('sortingChart');
                if (canvas && canvas.parentElement) {
                    const container = canvas.parentElement;
                    const newWidth = container.clientWidth - 30;
                    const newHeight = container.clientHeight - 30;
                    if (canvas.width !== newWidth || canvas.height !== newHeight) {
                        canvas.width = newWidth;
                        canvas.height = newHeight;
                        sortingChart.resize();
                    }
                }
                
                // Force chart update
                sortingChart.update();
                console.log('Chart updated successfully');
            } else {
                console.warn('Invalid graph data:', data);
            }
        })
        .catch(error => {
            console.error('Error fetching graph data:', error);
        });
}

// ImageBitmap + WebCodecs streaming for cameras (optimized for low lag)
function initCameraStream(canvasId, statusId, frameEndpoint, cameraName) {
    const canvas = document.getElementById(canvasId);
    const status = document.getElementById(statusId);
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    
    if (!canvas || !ctx) {
        console.error(`Canvas ${canvasId} not found`);
        return;
    }
    
    let frameCount = 0;
    let lastFrameTime = performance.now();
    let isStreaming = false;
    let pendingFrame = null;
    let isProcessing = false;
    
    // Use requestAnimationFrame for smooth rendering
    function renderFrame() {
        if (pendingFrame && !isProcessing) {
            isProcessing = true;
            const imageBitmap = pendingFrame;
            pendingFrame = null;
            
            // Resize canvas to fit container while maintaining aspect ratio
            const container = canvas.parentElement;
            const containerWidth = container.clientWidth;
            const containerHeight = container.clientHeight;
            
            if (containerWidth > 0 && containerHeight > 0) {
                const imageAspect = imageBitmap.width / imageBitmap.height;
                const containerAspect = containerWidth / containerHeight;
                
                let displayWidth, displayHeight;
                let offsetX = 0, offsetY = 0;
                
                if (imageAspect > containerAspect) {
                    // Image is wider - fit to width, center vertically
                    displayWidth = containerWidth;
                    displayHeight = containerWidth / imageAspect;
                    offsetY = (containerHeight - displayHeight) / 2;
                } else {
                    // Image is taller - fit to height, center horizontally
                    displayHeight = containerHeight;
                    displayWidth = containerHeight * imageAspect;
                    offsetX = (containerWidth - displayWidth) / 2;
                }
                
                // Set canvas size to fill container (use device pixel ratio for crisp rendering)
                const dpr = window.devicePixelRatio || 1;
                const canvasWidth = containerWidth * dpr;
                const canvasHeight = containerHeight * dpr;
                
                // Only resize if dimensions changed
                if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
                    canvas.width = canvasWidth;
                    canvas.height = canvasHeight;
                }
                
                // Set CSS size to match container (important for proper display)
                canvas.style.width = containerWidth + 'px';
                canvas.style.height = containerHeight + 'px';
                
                // Clear canvas first
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Scale context for device pixel ratio
                ctx.scale(dpr, dpr);
                
                // Draw ImageBitmap to canvas (scaled to fit, centered)
                ctx.drawImage(imageBitmap, offsetX, offsetY, displayWidth, displayHeight);
            } else {
                // Fallback: use image dimensions
                canvas.width = imageBitmap.width;
                canvas.height = imageBitmap.height;
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(imageBitmap, 0, 0);
            }
            
            // Clean up ImageBitmap
            imageBitmap.close();
            
            frameCount++;
            const now = performance.now();
            const fps = Math.round(1000 / (now - lastFrameTime));
            lastFrameTime = now;
            
            if (!isStreaming) {
                isStreaming = true;
                status.textContent = `Streaming (${fps} FPS)`;
                status.style.color = '#00ff00';
            } else if (frameCount % 30 === 0) {
                status.textContent = `Streaming (${fps} FPS)`;
            }
            
            isProcessing = false;
        }
        
        requestAnimationFrame(renderFrame);
    }
    
    // Start render loop
    requestAnimationFrame(renderFrame);
    
    async function fetchFrame() {
        try {
            const response = await fetch(`${frameEndpoint}?t=${Date.now()}`, {
                cache: 'no-cache',
                priority: 'high'
            });
            
            if (response.ok) {
                const blob = await response.blob();
                
                // Create ImageBitmap from blob (hardware accelerated)
                const imageBitmap = await createImageBitmap(blob);
                
                // Skip frame if still processing previous one (frame dropping for lower lag)
                if (!isProcessing && !pendingFrame) {
                    pendingFrame = imageBitmap;
                } else {
                    // Drop frame to reduce lag
                    imageBitmap.close();
                }
            } else if (response.status === 204) {
                if (isStreaming) {
                    status.textContent = 'Waiting for camera...';
                    status.style.color = '#ffaa00';
                    isStreaming = false;
                }
            }
        } catch (error) {
            console.error(`${cameraName} fetch error:`, error);
            if (isStreaming) {
                status.textContent = 'Connection error';
                status.style.color = '#ff0000';
                isStreaming = false;
            }
        }
        
        // Higher frame rate for smoother video (~25 FPS)
        setTimeout(fetchFrame, 40); // ~25 FPS
    }
    
    // Handle window resize to adjust canvas size
    function handleResize() {
        if (pendingFrame) {
            // Trigger a re-render with new size
            const container = canvas.parentElement;
            const containerWidth = container.clientWidth;
            const containerHeight = container.clientHeight;
            
            if (containerWidth > 0 && containerHeight > 0) {
                // Canvas will be resized in renderFrame
            }
        }
    }
    
    window.addEventListener('resize', handleResize);
    
    // Check for ImageBitmap support
    if (!('createImageBitmap' in window)) {
        status.textContent = 'ImageBitmap not supported';
        status.style.color = '#ff0000';
        return;
    }
    
    status.textContent = 'Connecting...';
    status.style.color = '#00aaff';
    
    // Start the fetch loop
    fetchFrame();
}

// Initialize camera streams
function setupCameraStreams() {
    initCameraStream('camera1', 'status1', '/api/frame1/raw', 'Camera 1');
    initCameraStream('camera2', 'status2', '/api/frame2/raw', 'Camera 2');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initChart();
    updateDatabase();
    updateStatistics();
    updateGraph();
    setupCameraStreams(); // Use ImageBitmap + WebCodecs streaming
    
    // Update database and statistics every 2 seconds (fresh data from PostgreSQL)
    setInterval(() => {
        updateDatabase();
        updateStatistics();
    }, 2000);
    
    // Update graph every 3 seconds (fresh data from PostgreSQL)
    setInterval(() => {
        updateGraph();
    }, 3000);
    
    // Initial load
    updateDatabase();
    updateStatistics();
    updateGraph();
});

