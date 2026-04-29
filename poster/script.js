document.addEventListener('DOMContentLoaded', () => {
    const btnTranslate = document.getElementById('btn-translate');
    const inputEs = document.getElementById('input-es');
    const outputEn = document.getElementById('output-en');
    
    // Animation elements
    const animEncoder = document.getElementById('anim-encoder');
    const animAttention = document.getElementById('anim-attention');
    const animDecoder = document.getElementById('anim-decoder');
    
    const tokenEncoder = document.getElementById('encoder-tokens');
    const tokenDecoder = document.getElementById('decoder-tokens');
    const attentionMatrix = document.getElementById('attention-matrix');

    // Create dot matrix for attention
    function initAttentionDots() {
        attentionMatrix.innerHTML = '';
        for(let i=0; i<10; i++){
            let d = document.createElement('div');
            d.className = 'att-node';
            attentionMatrix.appendChild(d);
        }
    }
    
    initAttentionDots();

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    async function triggerValidationAnimation(inputText, delayMs) {
        // Reset states
        animEncoder.classList.remove('active');
        animAttention.classList.remove('active');
        animDecoder.classList.remove('active');
        tokenEncoder.innerText = "...";
        tokenDecoder.innerText = "...";
        initAttentionDots();
        
        let words = inputText.trim().split(/\s+/).slice(0, 10);
        
        // 1. Encoder receives tokens
        animEncoder.classList.add('active');
        tokenEncoder.innerText = `[ ${words.join(' ')} ]`;
        await sleep(delayMs / 3);
        
        // 2. Attention processes weights
        animEncoder.classList.remove('active');
        animAttention.classList.add('active');
        // pulse dot colors
        let dots = attentionMatrix.querySelectorAll('.att-node');
        dots.forEach(d => {
            d.style.background = `rgba(248, 198, 48, ${Math.random()})`;
        });
        await sleep(delayMs / 3);
        
        // 3. Decoder generation phase
        animAttention.classList.remove('active');
        animDecoder.classList.add('active');
        tokenDecoder.innerText = "Generando secuencia autoregresiva...";
        await sleep(delayMs / 3);
        
        animDecoder.classList.remove('active');
    }

    btnTranslate.addEventListener('click', async () => {
        const text = inputEs.value.trim();
        if (!text) return;
        
        // Block UI and clear output
        btnTranslate.disabled = true;
        btnTranslate.innerText = "Traduciendo...";
        outputEn.value = "";
        
        // In local environments, the backend processing time is short but we'll pad the visualizer
        // Start animation loop while bringing data
        let animPromise = triggerValidationAnimation(text, 1500); 
        
        let translatedText = "Error al conectar con la API.";
        try {
            const result = await fetch('http://127.0.0.1:8000/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: text,
                    source_lang: "es",
                    target_lang: "en"
                })
            });
            if (result.ok) {
                const data = await result.json();
                translatedText = data.translation;
            } else {
                const errorData = await result.json();
                translatedText = `Error: ${errorData.detail || result.statusText}`;
            }
        } catch (error) {
            console.error("API error:", error);
            translatedText = "Error: El servidor (API) no está ejecutándose en http://127.0.0.1:8000.";
        }
        
        // Await animation finishing
        await animPromise;
        
        // Show result
        tokenDecoder.innerText = `[ ${translatedText.split(/\s+/).slice(0, 10).join(' ')} ]`;
        animDecoder.classList.add('active');
        
        outputEn.value = translatedText;
        
        btnTranslate.disabled = false;
        btnTranslate.innerText = "Traducir al Inglés ➔";
        
        setTimeout(() => animDecoder.classList.remove('active'), 1000);
    });
});
