const API_URL = "http://localhost:8000/api";

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
    });
});

async function runLesson1() {
    const input = document.getElementById('input-1');
    const history = document.getElementById('chat-history-1');
    const text = input.value;
    if (!text) return;

    appendMessage(history, text, 'user');
    input.value = 'Thinking...';
    input.disabled = true;

    try {
        const response = await fetch(`${API_URL}/lesson1`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });
        const data = await response.json();
        appendMessage(history, data.response, 'assistant');
    } catch (e) {
        appendMessage(history, "Error connecting to backend: " + e.message, 'system');
    } finally {
        input.value = '';
        input.disabled = false;
        input.focus();
    }
}

async function runLesson2() {
    const input = document.getElementById('input-2');
    const history = document.getElementById('output-2');
    const text = input.value;

    appendMessage(history, `Starting research on: ${text}`, 'user');

    try {
        const response = await fetch(`${API_URL}/lesson2`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });
        const data = await response.json();
        appendMessage(history, data.response, 'assistant');
    } catch (e) {
        appendMessage(history, "Error: " + e.message, 'system');
    }
}

async function runLesson3() {
    const input = document.getElementById('input-3');
    const history = document.getElementById('chat-history-3');
    const text = input.value;
    if (!text) return;

    appendMessage(history, text, 'user');
    input.value = 'Checking database...';
    input.disabled = true;

    try {
        const response = await fetch(`${API_URL}/lesson3`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });
        // Lesson 3 returns {"response": "text"} from our specialized endpoint
        const data = await response.json();
        appendMessage(history, data.response, 'assistant');
    } catch (e) {
        appendMessage(history, "Error: " + e.message, 'system');
    } finally {
        input.value = '';
        input.disabled = false;
        input.focus();
    }
}

function appendMessage(container, text, type) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}
