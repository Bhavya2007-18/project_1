// Synthetix Web Dashboard Application Engine

document.addEventListener('DOMContentLoaded', () => {
    // ----------------- TAB NAVIGATION -----------------
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const tabTitle = document.getElementById('current-tab-title');
    const tabDesc = document.getElementById('current-tab-desc');

    const tabMetadata = {
        overview: {
            title: "Pipeline Overview",
            desc: "End-to-End Deep Learning Text-to-Image Pipeline Sandbox"
        },
        tokenizer: {
            title: "Text Tokenizer & Embeddings",
            desc: "Convert natural language descriptions into subword tokens and multi-dimensional embeddings"
        },
        cgan: {
            title: "Conditional GAN for Geometric Shapes",
            desc: "Introduce labels as conditional parameters to direct generative synthesizers"
        },
        attention: {
            title: "Attention Guidance Sandbox",
            desc: "Explore how self-attention maps space and cross-attention maps words to visual targets"
        },
        dataset: {
            title: "Public Dataset Explorer",
            desc: "Inspect structures, resolutions, and text-image statistical distributions of public corpora"
        },
        refinement: {
            title: "Parameter-Efficient Fine-Tuning (LoRA)",
            desc: "Refine pre-trained models on target domain-specific visual distributions"
        }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            
            // Toggle active menu items
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Toggle active tab content
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(`pane-${tabId}`).classList.add('active');
            
            // Update Header Meta
            if (tabMetadata[tabId]) {
                tabTitle.textContent = tabMetadata[tabId].title;
                tabDesc.textContent = tabMetadata[tabId].desc;
            }
            
            // Handle specific tab initializations if required
            if (tabId === 'cgan') {
                renderCGAN();
            } else if (tabId === 'attention') {
                initAttentionTab();
            } else if (tabId === 'dataset') {
                initDatasetTab();
            }
        });
    });

    // ----------------- CLIENT-SIDE TOKENIZER SIMULATION -----------------
    const tokenizeText = (text) => {
        const clean = text.toLowerCase().trim();
        const words = clean.split(/\s+/).filter(w => w.length > 0);
        
        // Mock subword token mapping (vocabulary size 49408 like CLIP)
        const tokens = ["<|startoftext|>"];
        const tokenIds = [49406]; // Start token ID
        
        words.forEach(word => {
            // Simulate subword split for longer words
            if (word.length > 6) {
                const part1 = word.slice(0, 4);
                const part2 = "##" + word.slice(4);
                
                tokens.push(part1, part2);
                tokenIds.push(hashString(part1) % 40000 + 1000, hashString(part2) % 40000 + 1000);
            } else {
                tokens.push(word);
                tokenIds.push(hashString(word) % 40000 + 1000);
            }
        });
        
        tokens.push("<|endoftext|>");
        tokenIds.push(49407); // End token ID
        
        // Create CLIP standard sequence input of 77 tokens
        const sequenceLength = 77;
        const paddedIds = [...tokenIds];
        const attentionMask = Array(sequenceLength).fill(0);
        
        for (let i = 0; i < paddedIds.length && i < sequenceLength; i++) {
            attentionMask[i] = 1;
        }
        
        while (paddedIds.length < sequenceLength) {
            paddedIds.push(0);
        }
        
        return {
            tokens: tokens.slice(1, -1), // Return main tokens without start/end tags
            fullTokens: tokens,
            tokenIds: paddedIds,
            attentionMask: attentionMask,
            wordCount: words.length
        };
    };

    // FNV-1a Hash function for deterministic integer ids
    const hashString = (str) => {
        let hash = 2166136261;
        for (let i = 0; i < str.length; i++) {
            hash ^= str.charCodeAt(i);
            hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
        }
        return Math.abs(hash);
    };

    // Render 512-dim embedding slice visualization
    const renderEmbeddingVisual = (containerEl, promptText) => {
        containerEl.innerHTML = '';
        const seed = hashString(promptText || "default");
        
        // Generate a grid of colored pixels representing normalized latent dimensions
        for (let i = 0; i < 96; i++) { // display 96 cells as a sample slice
            const x = Math.sin(seed + i) * 10000;
            const val = x - Math.floor(x); // float between 0 and 1
            const cell = document.createElement('div');
            cell.className = 'embedding-pixel';
            
            // Use HSL for a highly futuristic purple/cyan distribution matching prompt seed
            const hue = val > 0.5 ? 270 : 180; // purple or cyan
            const saturation = 70 + Math.floor(val * 30);
            const lightness = 20 + Math.floor(val * 40);
            
            cell.style.backgroundColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            cell.title = `Dim ${i+1}: ${(val * 2 - 1).toFixed(4)}`;
            containerEl.appendChild(cell);
        }
    };

    // Setup Tokenizer Tab event listeners
    const btnTokenize = document.getElementById('btn-tokenize');
    const tokInputText = document.getElementById('tok-input-text');
    const tokStatCount = document.getElementById('tok-stat-count');
    const tokDisplaySubwords = document.getElementById('tok-display-subwords');
    const tokDisplayIds = document.getElementById('tok-display-ids');
    const tokDisplayEmbedding = document.getElementById('tok-display-embedding');

    const handleTokenizerAction = () => {
        const text = tokInputText.value;
        const result = tokenizeText(text);
        
        tokStatCount.textContent = result.tokens.length;
        
        // Render subword badges
        tokDisplaySubwords.innerHTML = '';
        result.tokens.forEach((tok, idx) => {
            const badge = document.createElement('div');
            badge.className = 'subword-item';
            
            const txt = document.createElement('span');
            txt.className = 'subword-text';
            txt.textContent = tok;
            
            const id = document.createElement('span');
            id.className = 'subword-id';
            id.textContent = `#${result.tokenIds[idx + 1]}`;
            
            badge.appendChild(txt);
            badge.appendChild(id);
            tokDisplaySubwords.appendChild(badge);
        });
        
        // Render input IDs cells
        tokDisplayIds.innerHTML = '';
        result.tokenIds.forEach((id, idx) => {
            const cell = document.createElement('div');
            cell.className = 'tensor-cell';
            if (id > 0 && idx < result.tokens.length + 2) {
                cell.classList.add('active-cell');
            }
            cell.textContent = id;
            tokDisplayIds.appendChild(cell);
        });
        
        // Render embedding grid
        renderEmbeddingVisual(tokDisplayEmbedding, text);
    };

    btnTokenize.addEventListener('click', handleTokenizerAction);
    // Initial run
    handleTokenizerAction();


    // ----------------- TAB 3: SHAPE CGAN SIMULATOR -----------------
    const cganCanvas = document.getElementById('cgan-canvas');
    const cganCtx = cganCanvas.getContext('2d');
    const cganNoiseSliders = document.querySelectorAll('.noise-slider');
    const cganNoiseFloorSlider = document.getElementById('cgan-noise-floor');
    const btnCganRandomize = document.getElementById('btn-cgan-randomize');
    const cganSpecZ = document.getElementById('cgan-spec-z');
    const cganSpecCond = document.getElementById('cgan-spec-cond');

    const getSelectedShape = () => {
        return document.querySelector('input[name="cgan-class"]:checked').value;
    };

    // Draw procedural shapes with simulated GAN grain
    const renderCGAN = () => {
        const shape = getSelectedShape();
        const n1 = parseFloat(document.getElementById('cgan-noise-1').value);
        const n2 = parseFloat(document.getElementById('cgan-noise-2').value);
        const n3 = parseFloat(document.getElementById('cgan-noise-3').value);
        const noiseFloor = parseFloat(cganNoiseFloorSlider.value);
        
        // Clear canvas
        cganCtx.fillStyle = '#000000';
        cganCtx.fillRect(0, 0, cganCanvas.width, cganCanvas.height);
        
        // Draw the main target conditional shape on an offscreen canvas
        const offCanvas = document.createElement('canvas');
        offCanvas.width = cganCanvas.width;
        offCanvas.height = cganCanvas.height;
        const offCtx = offCanvas.getContext('2d');
        offCtx.fillStyle = '#000000';
        offCtx.fillRect(0, 0, offCanvas.width, offCanvas.height);
        
        // Shape style properties altered by latent noise variables
        // Latent 1: Size/Scale modifier
        const sizeOffset = n1 * 12;
        const baseSize = 70 + sizeOffset;
        
        // Latent 2: Rotation angle
        const angle = n2 * 0.5; // radians
        
        // Latent 3: Shape eccentricity (aspect ratio skew)
        const skewX = 1 + n3 * 0.15;
        
        offCtx.save();
        offCtx.translate(offCanvas.width / 2, offCanvas.height / 2);
        offCtx.rotate(angle);
        offCtx.scale(skewX, 2 - skewX); // keep area roughly constant
        
        offCtx.fillStyle = '#ffffff';
        offCtx.strokeStyle = '#ffffff';
        offCtx.lineWidth = 12;
        
        if (shape === 'circle') {
            offCtx.beginPath();
            offCtx.arc(0, 0, baseSize / 2, 0, Math.PI * 2);
            offCtx.fill();
        } else if (shape === 'square') {
            offCtx.fillRect(-baseSize / 2, -baseSize / 2, baseSize, baseSize);
        } else if (shape === 'triangle') {
            offCtx.beginPath();
            offCtx.moveTo(0, -baseSize / 2);
            offCtx.lineTo(-baseSize / 2, baseSize / 2);
            offCtx.lineTo(baseSize / 2, baseSize / 2);
            offCtx.closePath();
            offCtx.fill();
        } else if (shape === 'star') {
            const spikes = 5;
            const outerRadius = baseSize / 2;
            const innerRadius = baseSize / 4;
            
            let rot = Math.PI / 2 * 3;
            let cx = 0;
            let cy = 0;
            let step = Math.PI / spikes;

            offCtx.beginPath();
            offCtx.moveTo(cx, cy - outerRadius);
            for (let i = 0; i < spikes; i++) {
                cx = Math.cos(rot) * outerRadius;
                cy = Math.sin(rot) * outerRadius;
                offCtx.lineTo(cx, cy);
                rot += step;

                cx = Math.cos(rot) * innerRadius;
                cy = Math.sin(rot) * innerRadius;
                offCtx.lineTo(cx, cy);
                rot += step;
            }
            offCtx.lineTo(0, -outerRadius);
            offCtx.closePath();
            offCtx.fill();
        }
        offCtx.restore();
        
        // Combine shape + simulated pixelated neural noise mapping
        const imgData = offCtx.getImageData(0, 0, cganCanvas.width, cganCanvas.height);
        const data = imgData.data;
        
        const finalImgData = cganCtx.createImageData(cganCanvas.width, cganCanvas.height);
        const finalData = finalImgData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            // Grayscale base intensity from offscreen drawing
            const baseIntensity = data[i]; // 0 or 255
            
            // Add gaussian-like simulated noise
            const randomNoise = (Math.random() - 0.5) * noiseFloor * 255;
            let intensity = baseIntensity + randomNoise;
            
            // GAN texture: blur edges or introduce mild grid artifact
            const xPixel = (i / 4) % cganCanvas.width;
            const yPixel = Math.floor((i / 4) / cganCanvas.width);
            const gridArtifact = Math.sin(xPixel * 0.8) * Math.cos(yPixel * 0.8) * 8 * noiseFloor;
            
            intensity += gridArtifact;
            intensity = Math.max(0, Math.min(255, intensity));
            
            // Tint shape cyan or purple based on latent parameter n1
            if (baseIntensity > 100) {
                // Inside shape
                if (n1 > 0) {
                    finalData[i] = intensity * 0.8;       // R (slight dim)
                    finalData[i + 1] = intensity * 0.95;   // G
                    finalData[i + 2] = intensity;          // B (stronger blue-ish cyan)
                } else {
                    finalData[i] = intensity;              // R (stronger purple)
                    finalData[i + 1] = intensity * 0.7;    // G
                    finalData[i + 2] = intensity * 0.95;   // B
                }
            } else {
                // Background dark noise
                finalData[i] = intensity * 0.1;
                finalData[i + 1] = intensity * 0.12;
                finalData[i + 2] = intensity * 0.2;
            }
            finalData[i + 3] = 255; // fully opaque
        }
        
        cganCtx.putImageData(finalImgData, 0, 0);
        
        // Update Specifications details text
        cganSpecCond.textContent = shape;
        cganSpecZ.textContent = `[${n1.toFixed(1)}, ${n2.toFixed(1)}, ${n3.toFixed(1)}]ᵀ`;
    };

    // Attach CGAN event listeners
    document.querySelectorAll('input[name="cgan-class"]').forEach(radio => {
        radio.addEventListener('change', renderCGAN);
    });
    
    cganNoiseSliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            document.getElementById(`${e.target.id}-val`).textContent = parseFloat(e.target.value).toFixed(1);
            renderCGAN();
        });
    });
    
    cganNoiseFloorSlider.addEventListener('input', (e) => {
        document.getElementById('cgan-noise-floor-val').textContent = parseFloat(e.target.value).toFixed(2);
        renderCGAN();
    });
    
    btnCganRandomize.addEventListener('click', () => {
        cganNoiseSliders.forEach(slider => {
            const randomVal = (Math.random() * 6 - 3).toFixed(1); // -3.0 to 3.0
            slider.value = randomVal;
            document.getElementById(`${slider.id}-val`).textContent = randomVal;
        });
        renderCGAN();
    });


    // ----------------- TAB 4: ATTENTION MAPS SIMULATOR -----------------
    const attnPromptSelect = document.getElementById('attn-prompt-select');
    const attnTokenHoverList = document.getElementById('attn-token-hover-list');
    const attnImageCanvas = document.getElementById('attn-image-canvas');
    const attnCtx = attnImageCanvas.getContext('2d');
    const attnOverlayMap = document.getElementById('attn-overlay-map');
    const attnActiveWord = document.getElementById('attn-active-word');
    const attnMaxWeight = document.getElementById('attn-max-weight');

    const scenes = {
        'scen-rose': {
            drawBg: (ctx) => {
                // Blue sky & sun
                const skyGrad = ctx.createLinearGradient(0, 0, 0, 150);
                skyGrad.addColorStop(0, '#1d4ed8'); // deep blue
                skyGrad.addColorStop(1, '#60a5fa'); // light sky blue
                ctx.fillStyle = skyGrad;
                ctx.fillRect(0, 0, 300, 300);
                
                // Yellow sun
                ctx.fillStyle = '#fef08a';
                ctx.shadowColor = '#facc15';
                ctx.shadowBlur = 15;
                ctx.beginPath();
                ctx.arc(250, 40, 24, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0; // reset
                
                // Green grass
                const grassGrad = ctx.createLinearGradient(0, 180, 0, 300);
                grassGrad.addColorStop(0, '#15803d');
                grassGrad.addColorStop(1, '#14532d');
                ctx.fillStyle = grassGrad;
                ctx.fillRect(0, 180, 300, 120);
            },
            drawFg: (ctx) => {
                // Red rose petals
                ctx.save();
                ctx.translate(150, 170);
                
                // Stem
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 5;
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.quadraticCurveTo(-10, 40, -5, 80);
                ctx.stroke();
                
                // Leaves
                ctx.fillStyle = '#15803d';
                ctx.beginPath();
                ctx.ellipse(-12, 35, 12, 6, -Math.PI/6, 0, Math.PI*2);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(10, 50, 10, 5, Math.PI/6, 0, Math.PI*2);
                ctx.fill();
                
                // Flower buds (overlapping ellipses)
                ctx.fillStyle = '#dc2626'; // dark red
                ctx.beginPath();
                ctx.ellipse(-16, -16, 20, 16, Math.PI/4, 0, Math.PI*2);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(16, -16, 20, 16, -Math.PI/4, 0, Math.PI*2);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(0, -26, 22, 18, 0, 0, Math.PI*2);
                ctx.fill();
                
                // Outer petals glow
                ctx.fillStyle = '#f87171'; // light red
                ctx.beginPath();
                ctx.arc(0, -10, 15, 0, Math.PI*2);
                ctx.fill();
                
                ctx.restore();
            },
            regions: {
                "sky": { x: 100, y: 50, r: 80 },
                "blue": { x: 150, y: 70, r: 100 },
                "sun": { x: 250, y: 40, r: 50 },
                "rose": { x: 150, y: 150, r: 50 },
                "red": { x: 150, y: 150, r: 45 },
                "grass": { x: 150, y: 240, r: 80 },
                "green": { x: 150, y: 240, r: 80 }
            }
        },
        'scen-island': {
            drawBg: (ctx) => {
                // Sunset sky gradient
                const skyGrad = ctx.createLinearGradient(0, 0, 0, 160);
                skyGrad.addColorStop(0, '#f97316'); // orange
                skyGrad.addColorStop(0.6, '#f43f5e'); // rose
                skyGrad.addColorStop(1, '#c084fc'); // purple
                ctx.fillStyle = skyGrad;
                ctx.fillRect(0, 0, 300, 300);
                
                // Sea
                const seaGrad = ctx.createLinearGradient(0, 160, 0, 300);
                seaGrad.addColorStop(0, '#0d9488'); // teal
                seaGrad.addColorStop(1, '#115e59'); // dark turquoise
                ctx.fillStyle = seaGrad;
                ctx.fillRect(0, 160, 300, 140);
            },
            drawFg: (ctx) => {
                // Island sand mound
                ctx.fillStyle = '#fef08a'; // yellow sand
                ctx.beginPath();
                ctx.ellipse(150, 180, 80, 24, 0, 0, Math.PI*2);
                ctx.fill();
                
                // Palm trunk
                ctx.strokeStyle = '#78350f'; // brown
                ctx.lineWidth = 6;
                ctx.beginPath();
                ctx.moveTo(140, 180);
                ctx.quadraticCurveTo(130, 120, 110, 85);
                ctx.stroke();
                
                // Palm leaves
                ctx.fillStyle = '#065f46';
                ctx.save();
                ctx.translate(110, 85);
                for (let i = 0; i < 5; i++) {
                    ctx.rotate(Math.PI / 2.5);
                    ctx.beginPath();
                    ctx.ellipse(25, 0, 25, 6, 0, 0, Math.PI*2);
                    ctx.fill();
                }
                ctx.restore();
            },
            regions: {
                "sky": { x: 150, y: 60, r: 120 },
                "orange": { x: 150, y: 60, r: 90 },
                "water": { x: 150, y: 230, r: 80 },
                "turquoise": { x: 150, y: 230, r: 80 },
                "island": { x: 150, y: 180, r: 60 },
                "green": { x: 110, y: 80, r: 50 },
                "palm": { x: 110, y: 80, r: 50 },
                "trees": { x: 110, y: 80, r: 55 }
            }
        },
        'scen-fire': {
            drawBg: (ctx) => {
                // Black charcoal background
                ctx.fillStyle = '#090a0f';
                ctx.fillRect(0, 0, 300, 300);
            },
            drawFg: (ctx) => {
                // Fire flame gradient
                ctx.save();
                ctx.translate(150, 200);
                
                // Draw 3 layers of flame (outer orange, mid red, inner yellow)
                const drawFlameLayer = (scale, color) => {
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.moveTo(0, 40 * scale);
                    ctx.bezierCurveTo(-40 * scale, 10 * scale, -30 * scale, -60 * scale, 0, -90 * scale);
                    ctx.bezierCurveTo(30 * scale, -60 * scale, 40 * scale, 10 * scale, 0, 40 * scale);
                    ctx.closePath();
                    ctx.fill();
                };
                
                ctx.shadowColor = '#ea580c';
                ctx.shadowBlur = 25;
                drawFlameLayer(1.1, '#ea580c'); // outer
                ctx.shadowBlur = 0;
                drawFlameLayer(0.8, '#dc2626'); // red
                drawFlameLayer(0.5, '#facc15'); // inner yellow
                
                ctx.restore();
            },
            regions: {
                "fire": { x: 150, y: 140, r: 75 },
                "flames": { x: 150, y: 140, r: 75 },
                "orange": { x: 150, y: 140, r: 80 },
                "yellow": { x: 150, y: 160, r: 40 },
                "black": { x: 40, y: 40, r: 60 },
                "background": { x: 40, y: 40, r: 100 }
            }
        }
    };

    const drawScene = (sceneKey) => {
        const scene = scenes[sceneKey];
        if (!scene) return;
        
        attnCtx.clearRect(0, 0, attnImageCanvas.width, attnImageCanvas.height);
        scene.drawBg(attnCtx);
        scene.drawFg(attnCtx);
    };

    const generateCrossAttentionHeatmap = (word, sceneKey) => {
        const scene = scenes[sceneKey];
        if (!scene) return;
        
        // Clear previous overlay
        attnOverlayMap.style.opacity = '0';
        
        // Find matching region coordinates
        const cleanWord = word.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"");
        const region = scene.regions[cleanWord];
        
        if (region) {
            // Create radial gradient for attention heat map
            const overlayCanvas = document.createElement('canvas');
            overlayCanvas.width = 300;
            overlayCanvas.height = 300;
            const oCtx = overlayCanvas.getContext('2d');
            
            const grad = oCtx.createRadialGradient(
                region.x, region.y, 2,
                region.x, region.y, region.r
            );
            grad.addColorStop(0, 'rgba(255, 0, 127, 0.9)');  // Hot pink core
            grad.addColorStop(0.3, 'rgba(255, 140, 0, 0.6)'); // Orange glow
            grad.addColorStop(0.7, 'rgba(138, 43, 226, 0.2)'); // Violet fringe
            grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            oCtx.fillStyle = grad;
            oCtx.fillRect(0, 0, 300, 300);
            
            // Set overlay image background
            attnOverlayMap.style.backgroundImage = `url(${overlayCanvas.toDataURL()})`;
            attnOverlayMap.style.opacity = '1';
            
            // Update stats
            attnActiveWord.textContent = `"${cleanWord}"`;
            attnMaxWeight.textContent = (0.75 + Math.random() * 0.22).toFixed(2);
        } else {
            // General grid background noise attention (for filler/stop words)
            const overlayCanvas = document.createElement('canvas');
            overlayCanvas.width = 300;
            overlayCanvas.height = 300;
            const oCtx = overlayCanvas.getContext('2d');
            
            oCtx.fillStyle = 'rgba(138, 43, 226, 0.05)';
            oCtx.fillRect(0, 0, 300, 300);
            
            // Weak distributed dots
            oCtx.fillStyle = 'rgba(0, 255, 255, 0.15)';
            for (let i = 0; i < 8; i++) {
                const rx = Math.random() * 300;
                const ry = Math.random() * 300;
                oCtx.beginPath();
                oCtx.arc(rx, ry, 30 + Math.random()*20, 0, Math.PI*2);
                oCtx.fill();
            }
            
            attnOverlayMap.style.backgroundImage = `url(${overlayCanvas.toDataURL()})`;
            attnOverlayMap.style.opacity = '1';
            
            attnActiveWord.textContent = `"${cleanWord}" (Weak)`;
            attnMaxWeight.textContent = (0.12 + Math.random() * 0.15).toFixed(2);
        }
    };

    const initAttentionTab = () => {
        const selectedScene = attnPromptSelect.value;
        const promptText = attnPromptSelect.options[attnPromptSelect.selectedIndex].text;
        
        drawScene(selectedScene);
        
        // Parse words and populate interactive hover list
        attnTokenHoverList.innerHTML = '';
        const words = promptText.split(/\s+/);
        
        words.forEach(word => {
            const badge = document.createElement('span');
            badge.className = 'attn-hover-badge';
            badge.textContent = word;
            
            badge.addEventListener('mouseenter', () => {
                document.querySelectorAll('.attn-hover-badge').forEach(b => b.classList.remove('active-hover'));
                badge.classList.add('active-hover');
                generateCrossAttentionHeatmap(word, attnPromptSelect.value);
            });
            
            attnTokenHoverList.appendChild(badge);
        });
        
        // Hide overlay initially
        attnOverlayMap.style.opacity = '0';
        attnActiveWord.textContent = '-';
        attnMaxWeight.textContent = '0.00';
    };

    attnPromptSelect.addEventListener('change', initAttentionTab);


    // ----------------- TAB 5: PUBLIC DATASET EXPLORER -----------------
    const chartCaptionLength = document.getElementById('chart-caption-length');
    const galleryClassFilter = document.getElementById('gallery-class-filter');
    const datasetGallery = document.getElementById('dataset-gallery');

    // Create caption distribution bars
    const renderCaptionLengthChart = () => {
        chartCaptionLength.innerHTML = '';
        const barHeights = [40, 110, 85, 55, 20]; // Mock word distribution
        const barValues = ["420", "1214", "948", "602", "215"];
        
        barHeights.forEach((h, idx) => {
            const bar = document.createElement('div');
            bar.className = 'chart-bar';
            bar.style.height = `${h}%`;
            bar.setAttribute('data-val', barValues[idx]);
            chartCaptionLength.appendChild(bar);
        });
    };

    // Flower canvas painters for gallery
    const drawGalleryFlower = (canvas, flowerType) => {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, w, h);
        
        // Draw green stalks
        ctx.strokeStyle = '#065f46';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(w/2, h);
        ctx.quadraticCurveTo(w/2 - 10, h/2 + 20, w/2, h/2);
        ctx.stroke();
        
        // Petal colors
        const colors = {
            'pink-primrose': '#f472b6',
            'wild-geranium': '#c084fc',
            'fire-lily': '#fb923c',
            'globe-thistle': '#60a5fa'
        };
        const color = colors[flowerType] || '#f3f4f6';
        
        ctx.save();
        ctx.translate(w/2, h/2);
        
        // Draw petals
        ctx.fillStyle = color;
        const numPetals = flowerType === 'globe-thistle' ? 12 : 5;
        const petalSize = flowerType === 'globe-thistle' ? 15 : 22;
        
        for (let i = 0; i < numPetals; i++) {
            ctx.rotate((Math.PI * 2) / numPetals);
            ctx.beginPath();
            ctx.ellipse(18, 0, petalSize, petalSize * 0.6, 0, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Center pistil
        ctx.fillStyle = flowerType === 'pink-primrose' ? '#facc15' : '#ffffff';
        if (flowerType === 'globe-thistle') ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(0, 0, 10, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
    };

    const galleryData = [
        { id: 1, type: "pink-primrose", cat: "Primrose (Cat 1)", cap: "a delicate primrose with light pink flowers and yellow center", res: "500x500" },
        { id: 2, type: "wild-geranium", cat: "Geranium (Cat 5)", cap: "five purple-violet petals with thin dark veins along them", res: "640x480" },
        { id: 3, type: "fire-lily", cat: "Lily (Cat 12)", cap: "bright orange lily flower with spotted petals curling backwards", res: "500x600" },
        { id: 4, type: "globe-thistle", cat: "Thistle (Cat 24)", cap: "a round spiky purple globe thistle head on a tall green stem", res: "640x480" },
        { id: 5, type: "pink-primrose", cat: "Primrose (Cat 1)", cap: "five round pink petals surround a bright yellow center", res: "500x600" },
        { id: 6, type: "wild-geranium", cat: "Geranium (Cat 5)", cap: "small purple wild geranium blooming near green shrubs", res: "500x500" }
    ];

    const renderGallery = (filterType) => {
        datasetGallery.innerHTML = '';
        const filtered = filterType === 'all' 
            ? galleryData 
            : galleryData.filter(item => item.type === filterType);
            
        filtered.forEach(item => {
            const card = document.createElement('div');
            card.className = 'gallery-card';
            
            const imgBox = document.createElement('div');
            imgBox.className = 'gallery-image-box';
            
            const canvas = document.createElement('canvas');
            canvas.width = 160;
            canvas.height = 130;
            
            const badge = document.createElement('span');
            badge.className = 'gallery-cat-badge';
            badge.textContent = item.cat;
            
            imgBox.appendChild(canvas);
            imgBox.appendChild(badge);
            
            const details = document.createElement('div');
            details.className = 'gallery-details';
            
            const caption = document.createElement('p');
            caption.className = 'gallery-caption';
            caption.textContent = item.cap;
            
            const res = document.createElement('span');
            res.className = 'gallery-res';
            res.textContent = `Resolution: ${item.res} px`;
            
            details.appendChild(caption);
            details.appendChild(res);
            
            card.appendChild(imgBox);
            card.appendChild(details);
            datasetGallery.appendChild(card);
            
            // Draw flower procedurally
            drawGalleryFlower(canvas, item.type);
        });
    };

    const initDatasetTab = () => {
        renderCaptionLengthChart();
        renderGallery(galleryClassFilter.value);
    };

    galleryClassFilter.addEventListener('change', () => {
        renderGallery(galleryClassFilter.value);
    });


    // ----------------- TAB 6: PEFT / LORA REFINEMENT -----------------
    const btnStartRefinement = document.getElementById('btn-start-refinement');
    const refDomainSelect = document.getElementById('ref-domain-select');
    const consoleLogs = document.getElementById('console-logs');
    const consoleStatusText = document.getElementById('console-status-text');
    const lossPath = document.getElementById('loss-path');
    const compBaseCanvas = document.getElementById('comp-base-canvas');
    const compRefinedCanvas = document.getElementById('comp-refined-canvas');
    const compBaseLbl = document.getElementById('comp-base-lbl');
    const compRefinedLbl = document.getElementById('comp-refined-lbl');

    const drawBaseModelOutput = (canvas, domain) => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0d0e15';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (domain === 'art') {
            // Draw generic landscape (unrefined base model output)
            ctx.fillStyle = '#3b82f6'; // flat blue sky
            ctx.fillRect(0, 0, 160, 90);
            
            ctx.fillStyle = '#f59e0b'; // sun
            ctx.beginPath();
            ctx.arc(120, 30, 10, 0, Math.PI*2);
            ctx.fill();
            
            ctx.fillStyle = '#10b981'; // green hills
            ctx.beginPath();
            ctx.moveTo(0, 160);
            ctx.quadraticCurveTo(50, 80, 100, 110);
            ctx.quadraticCurveTo(130, 90, 160, 120);
            ctx.lineTo(160, 160);
            ctx.closePath();
            ctx.fill();
        } else {
            // Draw blurry circle scan (unrefined medical output)
            const radGrad = ctx.createRadialGradient(80, 80, 5, 80, 80, 50);
            radGrad.addColorStop(0, '#ffffff');
            radGrad.addColorStop(0.5, 'rgba(255,255,255,0.2)');
            radGrad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = radGrad;
            ctx.fillRect(0, 0, 160, 160);
        }
    };

    const drawRefinedModelOutput = (canvas, domain, epochProgress = 10) => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0d0e15';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Progress defines how much noise or refinement is shown
        const ratio = epochProgress / 10;
        
        if (domain === 'art') {
            // Draw thick textured brushstroke oil painting (Renaissance style)
            // Stained canvas texture
            ctx.fillStyle = '#3a2d21';
            ctx.fillRect(0, 0, 160, 160);
            
            // sky strokes
            ctx.fillStyle = `rgba(30, 64, 175, ${ratio})`;
            ctx.fillRect(5, 5, 150, 80);
            
            // Impasto strokes (sky clouds)
            ctx.fillStyle = `rgba(219, 234, 254, ${ratio * 0.8})`;
            ctx.fillRect(20, 20, 80, 10);
            ctx.fillRect(40, 28, 60, 12);
            
            // Sun strokes
            ctx.fillStyle = `rgba(251, 191, 36, ${ratio})`;
            ctx.beginPath();
            ctx.arc(120, 35, 14, 0, Math.PI*2);
            ctx.fill();
            
            // Mountain canvas texture (Renaissance dark palette)
            ctx.fillStyle = `rgba(13, 59, 39, ${ratio})`;
            ctx.beginPath();
            ctx.moveTo(5, 155);
            ctx.lineTo(70, 75);
            ctx.lineTo(110, 115);
            ctx.lineTo(155, 95);
            ctx.lineTo(155, 155);
            ctx.closePath();
            ctx.fill();
            
            // Add brush stroke highlights (using loops to draw random thick segments)
            ctx.strokeStyle = `rgba(245, 158, 11, ${ratio * 0.3})`;
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(70, 140);
            ctx.lineTo(100, 135);
            ctx.stroke();
            
        } else {
            // Draw clean high-contrast medical MRI brain scan with ventricles
            ctx.save();
            ctx.translate(80, 80);
            ctx.scale(ratio, ratio);
            
            // Outer skull line
            ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(0, 0, 60, 0, Math.PI*2);
            ctx.stroke();
            
            // Internal structures (ventricles)
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.beginPath();
            ctx.arc(0, 0, 45, 0, Math.PI*2);
            ctx.fill();
            
            // Detailed dark nodes
            ctx.fillStyle = '#0d0e15';
            ctx.beginPath();
            ctx.ellipse(-15, -10, 12, 18, Math.PI/4, 0, Math.PI*2);
            ctx.ellipse(15, -10, 12, 18, -Math.PI/4, 0, Math.PI*2);
            ctx.fill();
            
            ctx.restore();
            
            // Add scanlines
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            for (let y = 0; y < 160; y += 4) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(160, y);
                ctx.stroke();
            }
        }
    };

    const updateLossCurve = (lossHistory) => {
        if (lossHistory.length === 0) {
            lossPath.setAttribute('d', '');
            return;
        }
        
        // SVG size is 400x150
        // Padding: Left 40, Right 10 (effective width 350)
        // Bottom: 130, Top: 10 (effective height 120)
        
        let pathD = '';
        const stepX = 350 / 10; // 10 epochs
        
        lossHistory.forEach((loss, idx) => {
            const x = 40 + idx * stepX;
            // Loss range 0.0 to 2.5
            // Mapping 0 -> 130px, 2.5 -> 10px
            const y = 130 - (loss / 2.5) * 120;
            
            if (idx === 0) {
                pathD += `M ${x} ${y}`;
            } else {
                pathD += ` L ${x} ${y}`;
            }
        });
        
        lossPath.setAttribute('d', pathD);
    };

    // Fine-tuning loop simulator
    let trainingInterval = null;
    
    const startRefinementTraining = () => {
        if (trainingInterval) {
            clearInterval(trainingInterval);
        }
        
        const domain = refDomainSelect.value;
        const domainName = refDomainSelect.options[refDomainSelect.selectedIndex].text;
        
        consoleLogs.innerHTML = `[SYSTEM] Initializing LoRA training loop for: ${domainName}...\n`;
        consoleStatusText.textContent = "TRAINING";
        consoleStatusText.className = "console-status training";
        
        btnStartRefinement.disabled = true;
        btnStartRefinement.textContent = "Refining Model...";
        
        // Reset base comparison panels
        drawBaseModelOutput(compBaseCanvas, domain);
        drawRefinedModelOutput(compRefinedCanvas, domain, 0); // start at epoch 0 (noisy/empty)
        
        if (domain === 'art') {
            compBaseLbl.textContent = "Standard CGI Vector Elements";
            compRefinedLbl.textContent = "Impasto Thick Brushstrokes (Epoch 0)";
        } else {
            compBaseLbl.textContent = "Diffuse Unaligned Circle";
            compRefinedLbl.textContent = "High-Contrast Tissue Slices (Epoch 0)";
        }
        
        let epoch = 1;
        let loss = 2.38;
        const lossHistory = [loss];
        updateLossCurve(lossHistory);
        
        trainingInterval = setInterval(() => {
            if (epoch > 10) {
                clearInterval(trainingInterval);
                consoleLogs.innerHTML += `\n[SUCCESS] LoRA Adapter weights successfully exported!\n[SAVED] Path: 'custom_domain_data/refined_lora_weights.safetensors'\n[INFO] Parameters Trained: 3,145,728 (LoRA Rank 8)`;
                consoleStatusText.textContent = "FINISHED";
                consoleStatusText.className = "console-status";
                
                btnStartRefinement.disabled = false;
                btnStartRefinement.textContent = "Start Fine-Tuning Execution";
                return;
            }
            
            // Training simulation computations
            const epochLossReduction = 0.22 * (1.1 / epoch) + (Math.random() - 0.5) * 0.08;
            loss = Math.max(0.18, loss - epochLossReduction);
            lossHistory.push(loss);
            
            // Console print log
            consoleLogs.innerHTML += `Epoch ${epoch}/10 - Train Loss: ${loss.toFixed(4)} - val_loss: ${(loss * 1.05 + Math.random()*0.02).toFixed(4)} - Time/epoch: 1.4s\n`;
            consoleLogs.scrollTop = consoleLogs.scrollHeight;
            
            // Update loss path graph
            updateLossCurve(lossHistory);
            
            // Incrementally refine output canvas
            drawRefinedModelOutput(compRefinedCanvas, domain, epoch);
            compRefinedLbl.textContent = `${domain === 'art' ? 'Impasto Brushstrokes' : 'Ventricle Boundaries'} (Epoch ${epoch})`;
            
            epoch++;
        }, 1200);
    };

    btnStartRefinement.addEventListener('click', startRefinementTraining);
    
    // Initial static renders
    drawBaseModelOutput(compBaseCanvas, 'art');
    drawRefinedModelOutput(compRefinedCanvas, 'art', 0);


    // ----------------- TAB 1: INTEGRATED PIPELINE RUN ENGINE -----------------
    const btnRunPipeline = document.getElementById('btn-run-pipeline');
    const pipelinePrompt = document.getElementById('pipeline-prompt');
    const pipelineOutputCanvas = document.getElementById('pipeline-output-canvas');
    const pipelineCtx = pipelineOutputCanvas.getContext('2d');
    const pipelineAttnOverlay = document.getElementById('pipeline-attn-overlay');
    const pipelineActiveModel = document.getElementById('pipeline-active-model');
    const pipelineTokensContainer = document.getElementById('pipeline-tokens-container');
    const pipelineHint = document.getElementById('pipeline-hint');
    
    // Quick preset click handler
    document.querySelectorAll('.preset-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            pipelinePrompt.value = tag.getAttribute('data-prompt');
            runIntegratedPipeline();
        });
    });

    const setFlowStepState = (stepId, state) => {
        const el = document.getElementById(stepId);
        if (!el) return;
        el.className = 'flow-step';
        if (state) el.classList.add(state); // 'active' or 'completed'
    };

    const runIntegratedPipeline = () => {
        const prompt = pipelinePrompt.value.trim();
        if (!prompt) return;
        
        btnRunPipeline.disabled = true;
        btnRunPipeline.textContent = "Processing...";
        pipelineActiveModel.textContent = "-";
        pipelineTokensContainer.innerHTML = '<span class="no-tokens-placeholder">Processing text...</span>';
        pipelineHint.style.opacity = '0.3';
        pipelineAttnOverlay.style.opacity = '0';
        
        // Reset telemetry step states
        setFlowStepState('flow-tok', 'active');
        setFlowStepState('flow-emb', '');
        setFlowStepState('flow-route', '');
        setFlowStepState('flow-gen', '');
        
        document.getElementById('flow-tok-output').textContent = '-';
        document.getElementById('flow-emb-output').textContent = '-';
        document.getElementById('flow-route-output').textContent = '-';
        
        // Simulate step-by-step pipeline timings
        setTimeout(() => {
            // 1. Text processing completed
            const tokResult = tokenizeText(prompt);
            setFlowStepState('flow-tok', 'completed');
            setFlowStepState('flow-emb', 'active');
            document.getElementById('flow-tok-output').textContent = `Tokens: ${tokResult.tokens.join(', ')}`;
            
            setTimeout(() => {
                // 2. Embedding mapping completed
                setFlowStepState('flow-emb', 'completed');
                setFlowStepState('flow-route', 'active');
                document.getElementById('flow-emb-output').textContent = `Tensor shape: [1, 77, 512]`;
                
                setTimeout(() => {
                    // 3. Routing Completed
                    // Check if prompt contains shape label for CGAN
                    const shapeLabels = ["circle", "square", "triangle", "shape"];
                    let isShape = false;
                    let matchedLabel = "default";
                    
                    shapeLabels.forEach(lbl => {
                        if (prompt.toLowerCase().includes(lbl)) {
                            isShape = true;
                            matchedLabel = lbl === "shape" ? "circle" : lbl;
                        }
                    });
                    
                    setFlowStepState('flow-route', 'completed');
                    setFlowStepState('flow-gen', 'active');
                    
                    if (isShape) {
                        document.getElementById('flow-route-output').textContent = `Matched Shape label: '${matchedLabel}'. Route -> CGAN`;
                        pipelineActiveModel.textContent = "Shape CGAN Generator";
                    } else {
                        document.getElementById('flow-route-output').textContent = `Complex prompt. Route -> AttnGAN Engine`;
                        pipelineActiveModel.textContent = "Attention GAN (Self & Cross)";
                    }
                    
                    setTimeout(() => {
                        // 4. Generation completed
                        setFlowStepState('flow-gen', 'completed');
                        btnRunPipeline.disabled = false;
                        btnRunPipeline.textContent = "Run Pipeline";
                        
                        // Render result images
                        if (isShape) {
                            // CGAN procedural drawing
                            pipelineCtx.fillStyle = '#000000';
                            pipelineCtx.fillRect(0, 0, 256, 256);
                            
                            pipelineCtx.fillStyle = '#8a2be2'; // violet shape
                            pipelineCtx.strokeStyle = '#ffffff';
                            pipelineCtx.lineWidth = 6;
                            
                            pipelineCtx.beginPath();
                            if (matchedLabel === 'circle') {
                                pipelineCtx.arc(128, 128, 55, 0, Math.PI * 2);
                                pipelineCtx.fill();
                            } else if (matchedLabel === 'square') {
                                pipelineCtx.fillRect(73, 73, 110, 110);
                            } else {
                                // triangle
                                pipelineCtx.moveTo(128, 73);
                                pipelineCtx.lineTo(68, 178);
                                pipelineCtx.lineTo(188, 178);
                                pipelineCtx.closePath();
                                pipelineCtx.fill();
                            }
                            
                            // Add mild grain
                            const pData = pipelineCtx.getImageData(0, 0, 256, 256);
                            for (let idx = 0; idx < pData.data.length; idx += 4) {
                                const grain = (Math.random() - 0.5) * 12;
                                pData.data[idx] = Math.max(0, Math.min(255, pData.data[idx] + grain));
                                pData.data[idx+1] = Math.max(0, Math.min(255, pData.data[idx+1] + grain));
                                pData.data[idx+2] = Math.max(0, Math.min(255, pData.data[idx+2] + grain));
                            }
                            pipelineCtx.putImageData(pData, 0, 0);
                            
                            // Display single token for shape label
                            pipelineTokensContainer.innerHTML = '';
                            const tBadge = document.createElement('span');
                            tBadge.className = 'token-badge';
                            tBadge.textContent = matchedLabel;
                            pipelineTokensContainer.appendChild(tBadge);
                            pipelineHint.innerHTML = "Generated via Conditional GAN. Conditioned on shape: <strong>" + matchedLabel + "</strong>.";
                            pipelineHint.style.opacity = '0.8';
                            
                        } else {
                            // AttnGAN Complex text simulation
                            pipelineCtx.clearRect(0, 0, 256, 256);
                            
                            // Paint procedural scene based on text matching
                            let hasRose = prompt.toLowerCase().includes('rose') || prompt.toLowerCase().includes('flower');
                            
                            // Sky
                            const skyGrad = pipelineCtx.createLinearGradient(0, 0, 0, 130);
                            skyGrad.addColorStop(0, '#1d4ed8');
                            skyGrad.addColorStop(1, '#60a5fa');
                            pipelineCtx.fillStyle = skyGrad;
                            pipelineCtx.fillRect(0, 0, 256, 256);
                            
                            // Sun
                            pipelineCtx.fillStyle = '#fef08a';
                            pipelineCtx.beginPath();
                            pipelineCtx.arc(200, 40, 16, 0, Math.PI*2);
                            pipelineCtx.fill();
                            
                            // Ground
                            const groundGrad = pipelineCtx.createLinearGradient(0, 140, 0, 256);
                            groundGrad.addColorStop(0, '#16a34a');
                            groundGrad.addColorStop(1, '#14532d');
                            pipelineCtx.fillStyle = groundGrad;
                            pipelineCtx.fillRect(0, 140, 256, 116);
                            
                            if (hasRose) {
                                // Draw red rose stem/bud
                                pipelineCtx.strokeStyle = '#15803d';
                                pipelineCtx.lineWidth = 4;
                                pipelineCtx.beginPath();
                                pipelineCtx.moveTo(128, 200);
                                pipelineCtx.quadraticCurveTo(120, 150, 128, 125);
                                pipelineCtx.stroke();
                                
                                pipelineCtx.fillStyle = '#dc2626';
                                pipelineCtx.beginPath();
                                pipelineCtx.ellipse(128, 115, 18, 14, 0, 0, Math.PI*2);
                                pipelineCtx.fill();
                                pipelineCtx.fillStyle = '#ef4444';
                                pipelineCtx.beginPath();
                                pipelineCtx.arc(128, 110, 10, 0, Math.PI*2);
                                pipelineCtx.fill();
                            }
                            
                            // Render hover tokens
                            pipelineTokensContainer.innerHTML = '';
                            tokResult.tokens.forEach(tok => {
                                const badge = document.createElement('span');
                                badge.className = 'token-badge';
                                badge.textContent = tok;
                                
                                badge.addEventListener('mouseenter', () => {
                                    // Generate overlay mask coordinates based on keyword
                                    const cleanTok = tok.toLowerCase();
                                    const oCanvas = document.createElement('canvas');
                                    oCanvas.width = 256;
                                    oCanvas.height = 256;
                                    const oCtx = oCanvas.getContext('2d');
                                    
                                    let rx = 128, ry = 128, rr = 60;
                                    if (cleanTok === 'rose' || cleanTok === 'flower' || cleanTok === 'red') {
                                        rx = 128; ry = 115; rr = 40;
                                    } else if (cleanTok === 'sun' || cleanTok === 'yellow' || cleanTok === 'golden') {
                                        rx = 200; ry = 40; rr = 30;
                                    } else if (cleanTok === 'sky' || cleanTok === 'blue') {
                                        rx = 128; ry = 60; rr = 90;
                                    } else if (cleanTok === 'grass' || cleanTok === 'green') {
                                        rx = 128; ry = 200; rr = 80;
                                    }
                                    
                                    const grad = oCtx.createRadialGradient(rx, ry, 2, rx, ry, rr);
                                    grad.addColorStop(0, 'rgba(255, 0, 127, 0.9)');  // Hot pink core
                                    grad.addColorStop(0.3, 'rgba(255, 140, 0, 0.6)'); // Orange glow
                                    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
                                    oCtx.fillStyle = grad;
                                    oCtx.fillRect(0, 0, 256, 256);
                                    
                                    pipelineAttnOverlay.style.backgroundImage = `url(${oCanvas.toDataURL()})`;
                                    pipelineAttnOverlay.style.opacity = '1';
                                });
                                
                                badge.addEventListener('mouseleave', () => {
                                    pipelineAttnOverlay.style.opacity = '0';
                                });
                                
                                pipelineTokensContainer.appendChild(badge);
                            });
                            
                            pipelineHint.innerHTML = "Hover over the tokenized words above to view their corresponding spatial cross-attention heatmaps on the output canvas.";
                            pipelineHint.style.opacity = '0.8';
                        }
                    }, 1000);
                }, 800);
            }, 800);
        }, 800);
    };

    btnRunPipeline.addEventListener('click', runIntegratedPipeline);
    // Initial preview run
    runIntegratedPipeline();
});
