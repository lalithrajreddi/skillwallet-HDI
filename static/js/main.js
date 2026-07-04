document.addEventListener('DOMContentLoaded', () => {
    // ------------------ Tab Controls ------------------
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Remove active classes
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active classes
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // ------------------ Slider Syncing ------------------
    const inputsToSync = [
        { slider: 'life_expectancy_slider', number: 'life_expectancy' },
        { slider: 'expected_schooling_slider', number: 'expected_schooling' },
        { slider: 'mean_schooling_slider', number: 'mean_schooling' },
        { slider: 'gni_capita_slider', number: 'gni_capita' }
    ];

    inputsToSync.forEach(pair => {
        const sliderEl = document.getElementById(pair.slider);
        const numberEl = document.getElementById(pair.number);

        if (sliderEl && numberEl) {
            // Slider to Number input
            sliderEl.addEventListener('input', (e) => {
                numberEl.value = e.target.value;
            });

            // Number to Slider input
            numberEl.addEventListener('change', (e) => {
                let val = parseFloat(e.target.value);
                const min = parseFloat(sliderEl.min);
                const max = parseFloat(sliderEl.max);

                if (isNaN(val)) val = min;
                if (val < min) val = min;
                if (val > max) val = max;

                numberEl.value = val;
                sliderEl.value = val;
            });
        }
    });

    // ------------------ Prediction Form Handler ------------------
    const predForm = document.getElementById('prediction-form');
    const btnPredict = document.getElementById('btn-predict');
    const resPlaceholder = document.getElementById('res-placeholder');
    const resSuccess = document.getElementById('res-success-container');

    if (predForm) {
        predForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Show spinner and disable button
            const btnText = btnPredict.querySelector('.btn-text');
            const spinner = btnPredict.querySelector('.spinner');
            btnText.textContent = "Analyzing Indicators...";
            spinner.classList.remove('hidden');
            btnPredict.disabled = true;

            // Get values
            const life_expectancy = document.getElementById('life_expectancy').value;
            const expected_schooling = document.getElementById('expected_schooling').value;
            const mean_schooling = document.getElementById('mean_schooling').value;
            const gni_capita = document.getElementById('gni_capita').value;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        life_expectancy,
                        expected_schooling,
                        mean_schooling,
                        gni_capita
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    // Show success block, hide placeholder
                    resPlaceholder.classList.add('hidden');
                    resSuccess.classList.remove('hidden');

                    // Set HDI text
                    document.getElementById('res-hdi-score').textContent = data.predicted_hdi.toFixed(3);

                    // Set Tier Badge
                    const tierBadge = document.getElementById('res-tier-badge');
                    tierBadge.textContent = data.tier + " HDI";
                    
                    // Remove old tier classes and set new one
                    tierBadge.className = 'badge';
                    if (data.tier === 'Very High') tierBadge.classList.add('tier-very-high');
                    else if (data.tier === 'High') tierBadge.classList.add('tier-high');
                    else if (data.tier === 'Medium') tierBadge.classList.add('tier-medium');
                    else if (data.tier === 'Low') tierBadge.classList.add('tier-low');

                    // Animate Radial Progress
                    // Circle circumference is 2 * pi * r = 2 * 3.14159 * 50 = 314.159
                    const radialFill = document.getElementById('res-radial-fill');
                    const circumference = 314.159;
                    const offset = circumference - (data.predicted_hdi * circumference);
                    radialFill.style.strokeDashoffset = offset;

                    // Update Sub-index numbers & progress bars
                    updateProgressBar('health-idx', data.indices.health);
                    updateProgressBar('edu-idx', data.indices.education);
                    updateProgressBar('inc-idx', data.indices.income);

                    // Update Comparisons List
                    const compList = document.getElementById('res-comparisons-list');
                    compList.innerHTML = '';
                    
                    if (data.comparisons && data.comparisons.length > 0) {
                        data.comparisons.forEach(c => {
                            const compItem = document.createElement('div');
                            compItem.className = 'comp-item';
                            
                            // Map tier class
                            let tierClass = 'tier-low';
                            if (c.tier === 'Very High') tierClass = 'tier-very-high';
                            else if (c.tier === 'High') tierClass = 'tier-high';
                            else if (c.tier === 'Medium') tierClass = 'tier-medium';

                            compItem.innerHTML = `
                                <div class="comp-country-header">
                                    <span class="comp-name">${c.country}</span>
                                    <span class="comp-badge ${tierClass}">${c.hdi.toFixed(3)}</span>
                                </div>
                                <div class="comp-metrics">
                                    <span>Life Exp: <strong>${c.life_expectancy.toFixed(1)} yrs</strong></span>
                                    <span>GNI: <strong>$${c.gni.toLocaleString()}</strong></span>
                                </div>
                            `;
                            compList.appendChild(compItem);
                        });
                    } else {
                        compList.innerHTML = '<p class="comp-desc">No comparisons available.</p>';
                    }
                    
                    // Smooth scroll to results on small screens
                    if (window.innerWidth < 1024) {
                        document.getElementById('prediction-result-panel').scrollIntoView({ behavior: 'smooth' });
                    }

                } else {
                    alert('Error: ' + (data.error || 'Failed to generate prediction.'));
                }
            } catch (err) {
                console.error(err);
                alert('Prediction service failed to connect.');
            } finally {
                // Reset button state
                btnText.textContent = "Generate Prediction";
                spinner.classList.add('hidden');
                btnPredict.disabled = false;
            }
        });
    }

    function updateProgressBar(id, value) {
        const textEl = document.getElementById(`val-${id}`);
        const fillEl = document.getElementById(`fill-${id}`);
        if (textEl && fillEl) {
            textEl.textContent = value.toFixed(3);
            fillEl.style.width = `${value * 100}%`;
        }
    }

    // ------------------ Country Explorer Database ------------------
    const countrySearch = document.getElementById('country-search');
    const tierFilter = document.getElementById('tier-filter');
    const tableBody = document.getElementById('countries-table-body');
    const btnLoadCountries = document.getElementById('btn-load-countries');

    async function loadCountryDatabase() {
        if (!tableBody) return;
        
        tableBody.innerHTML = '<tr><td colspan="9" class="td-status">Loading country records...</td></tr>';
        
        const searchQuery = countrySearch.value.trim();
        const tierQuery = tierFilter.value;
        const url = `/api/countries?search=${encodeURIComponent(searchQuery)}&tier=${encodeURIComponent(tierQuery)}`;

        try {
            const response = await fetch(url);
            const countries = await response.json();
            
            tableBody.innerHTML = '';
            
            if (countries.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="9" class="td-status">No country records found matching your filters.</td></tr>';
                return;
            }

            countries.forEach(c => {
                const tr = document.createElement('tr');
                
                // Get tier badge class
                let tierClass = 'tier-low';
                if (c.tier === 'Very High') tierClass = 'tier-very-high';
                else if (c.tier === 'High') tierClass = 'tier-high';
                else if (c.tier === 'Medium') tierClass = 'tier-medium';

                tr.innerHTML = `
                    <td class="td-center font-semibold">${c.rank || '-'}</td>
                    <td class="font-medium text-white">${c.country}</td>
                    <td class="td-center text-slate">${c.iso || '-'}</td>
                    <td class="td-right">${c.life_expectancy.toFixed(1)}</td>
                    <td class="td-right">${c.expected_schooling.toFixed(1)}</td>
                    <td class="td-right">${c.mean_schooling.toFixed(1)}</td>
                    <td class="td-right font-mono">$${c.gni.toLocaleString()}</td>
                    <td class="td-right font-bold text-white">${c.hdi.toFixed(3)}</td>
                    <td><span class="badge ${tierClass}">${c.tier}</span></td>
                `;
                
                // Quick load row items to form if row is clicked for predicting
                tr.addEventListener('click', () => {
                    document.getElementById('life_expectancy').value = c.life_expectancy;
                    document.getElementById('life_expectancy_slider').value = c.life_expectancy;
                    
                    document.getElementById('expected_schooling').value = c.expected_schooling;
                    document.getElementById('expected_schooling_slider').value = c.expected_schooling;
                    
                    document.getElementById('mean_schooling').value = c.mean_schooling;
                    document.getElementById('mean_schooling_slider').value = c.mean_schooling;
                    
                    document.getElementById('gni_capita').value = c.gni;
                    document.getElementById('gni_capita_slider').value = c.gni > 100000 ? 100000 : c.gni;

                    // Switch tab to predictor
                    document.querySelector('[data-tab="predictor-tab"]').click();
                    
                    // Highlight the prediction form
                    predForm.classList.add('form-highlight');
                    setTimeout(() => predForm.classList.remove('form-highlight'), 1000);
                });

                tableBody.appendChild(tr);
            });
        } catch (err) {
            console.error(err);
            tableBody.innerHTML = '<tr><td colspan="9" class="td-status text-red">Failed to load country database records.</td></tr>';
        }
    }

    // Set up search event listeners with debouncing
    let searchDebounceTimeout;
    if (countrySearch) {
        countrySearch.addEventListener('input', () => {
            clearTimeout(searchDebounceTimeout);
            searchDebounceTimeout = setTimeout(loadCountryDatabase, 300);
        });
    }

    if (tierFilter) {
        tierFilter.addEventListener('change', loadCountryDatabase);
    }

    if (btnLoadCountries) {
        btnLoadCountries.addEventListener('click', loadCountryDatabase);
    }

    // ------------------ Global stats loading ------------------
    async function loadGlobalStats() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();

            if (response.ok) {
                // Header Global Average
                document.getElementById('lbl-global-hdi').textContent = stats.global_averages.hdi.toFixed(3);
                
                // If on About tab, populate model variables dynamically from server
                const coefList = document.getElementById('coef-list');
                if (coefList && stats.model_info && stats.model_info.coefficients) {
                    const c = stats.model_info.coefficients;
                    const intercept = stats.model_info.intercept;
                    coefList.innerHTML = `
                        <li><span class="coef-name">Life Expectancy</span> <span class="coef-val">${c.life_expectancy >= 0 ? '+' : ''}${c.life_expectancy.toFixed(6)}</span></li>
                        <li><span class="coef-name">Expected Years of Schooling</span> <span class="coef-val">${c.expected_schooling >= 0 ? '+' : ''}${c.expected_schooling.toFixed(6)}</span></li>
                        <li><span class="coef-name">Mean Years of Schooling</span> <span class="coef-val">${c.mean_schooling >= 0 ? '+' : ''}${c.mean_schooling.toFixed(6)}</span></li>
                        <li><span class="coef-name">GNI Per Capita</span> <span class="coef-val">${c.gni_capita >= 0 ? '+' : ''}${c.gni_capita.toFixed(8)}</span></li>
                        <li class="intercept-row"><span class="coef-name">Y-Intercept</span> <span class="coef-val">${intercept >= 0 ? '+' : ''}${intercept.toFixed(6)}</span></li>
                    `;
                }
            }
        } catch (err) {
            console.warn("Failed to load statistics on startup", err);
        }
    }

    loadGlobalStats();
    loadCountryDatabase(); // Initial table load

    // ------------------ Image Lightbox Modal ------------------
    const images = document.querySelectorAll('.plot-img');
    const lightbox = document.getElementById('image-lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxCaption = document.getElementById('lightbox-caption');
    const lightboxClose = document.getElementById('lightbox-close');

    images.forEach(img => {
        img.addEventListener('click', () => {
            lightbox.classList.add('active');
            lightboxImg.src = img.src;
            
            // Set caption from the sibling card caption
            const parent = img.closest('.plot-card');
            const caption = parent.querySelector('.plot-caption');
            if (caption) {
                lightboxCaption.innerHTML = caption.innerHTML;
            } else {
                lightboxCaption.textContent = img.alt;
            }
        });
    });

    if (lightboxClose) {
        lightboxClose.addEventListener('click', () => {
            lightbox.classList.remove('active');
        });
    }

    if (lightbox) {
        lightbox.addEventListener('click', (e) => {
            if (e.target !== lightboxImg && e.target !== lightboxCaption) {
                lightbox.classList.remove('active');
            }
        });
    }
});
