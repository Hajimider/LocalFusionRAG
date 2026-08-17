const $ = (id) => document.getElementById(id);
function setText(id, value) { $(id).textContent = value; }

async function refreshHealth() {
  try {
    const data = await (await fetch('/api/health')).json();
    setText('health', data.status === 'ok' ? '服务正常' : '服务异常');
    setText('index-state', data.index_ready ? '已建库' : '未建库');
    const provider = data.llm_provider === 'api' ? `API：${data.model_name}` : '本地 GGUF';
    setText('model-state', data.model_configured ? `${provider} · ${data.model_loaded ? '已加载' : '待加载'}` : `${provider} · 未配置`);
  } catch (_) {
    setText('health', '无法连接服务');
    setText('model-state', '请先启动 FastAPI');
  }
}

async function uploadFile() {
  const file = $('file-input').files[0];
  if (!file) { setText('index-message', '请选择一个文档。'); return; }
  const form = new FormData(); form.append('file', file); $('upload-button').disabled = true;
  try {
    const response = await fetch('/api/documents/upload', { method: 'POST', body: form });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || '上传失败');
    setText('index-message', `${data.filename} 已上传，请重建索引。`);
  } catch (error) { setText('index-message', error.message); }
  finally { $('upload-button').disabled = false; }
}

async function buildIndex() {
  $('build-button').disabled = true; setText('index-message', '正在解析文档并构建索引...');
  try {
    const response = await fetch('/api/index/build', { method: 'POST' });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || '建库失败');
    setText('index-message', `完成：${data.document_sections} 个文档区段，${data.chunks} 个检索片段。`); await refreshHealth();
  } catch (error) { setText('index-message', error.message); }
  finally { $('build-button').disabled = false; }
}

function addMessage(role, text = '') {
  document.querySelector('.empty-state')?.remove();
  const wrapper = document.createElement('article'); wrapper.className = `message ${role}`;
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? '用户' : '法律资料助手';
  const body = document.createElement('div'); body.className = 'message-body'; body.textContent = text; wrapper.append(label, body); $('messages').appendChild(wrapper); return { wrapper, body };
}

function addSources(wrapper, data) {
  const section = document.createElement('div'); section.className = 'sources';
  const info = document.createElement('p'); info.className = 'retrieval-info';
  info.textContent = `检索：${data.query} · ${data.mode} · 意图：${data.intent} · 链路：${data.generation_chain || 'qa'}`;
  section.appendChild(info);
  data.sources.forEach((source, index) => {
    const details = document.createElement('details'); details.className = 'source-item';
    const summary = document.createElement('summary'); summary.textContent = `[资料${index + 1}] ${source.file} · ${source.legal_category || source.doc_type} · ${source.validity}${source.case_number ? ` · ${source.case_number}` : ''}`;
    const excerpt = document.createElement('p'); excerpt.className = 'source-excerpt'; excerpt.textContent = source.excerpt; details.append(summary, excerpt); section.appendChild(details);
  });
  wrapper.appendChild(section);
}

async function askQuestion(event) {
  event.preventDefault(); const question = $('question').value.trim(); if (!question) return;
  addMessage('user', question); const assistant = addMessage('assistant'); $('question').value = ''; $('question').disabled = true; setText('chat-state', '生成中...');
  try {
    const response = await fetch('/api/chat/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
      question, use_rag: $('rag-toggle').checked, top_k: Number($('top-k').value), retrieval_mode: $('retrieval-mode').value,
      rewrite_query: $('rewrite-toggle').checked, rerank: $('rerank-toggle').checked, document_type: $('document-type').value, validity: $('validity').value,
    }) });
    if (!response.ok) { const error = await response.json(); throw new Error(error.detail || '问答失败'); }
    const reader = response.body.getReader(); const decoder = new TextDecoder('utf-8'); let buffer = '';
    while (true) {
      const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done }); const lines = buffer.split('\n'); buffer = lines.pop();
      for (const line of lines) { if (!line.trim()) continue; const eventData = JSON.parse(line); if (eventData.event === 'sources') addSources(assistant.wrapper, eventData); if (eventData.event === 'token') assistant.body.textContent += eventData.text; if (eventData.event === 'error') throw new Error(eventData.message); }
      if (done) break;
    }
    setText('chat-state', '就绪');
  } catch (error) { assistant.body.textContent = `请求失败：${error.message}`; setText('chat-state', '请求失败'); }
  finally { $('question').disabled = false; $('question').focus(); }
}

document.querySelectorAll('.example-prompt').forEach((button) => {
  button.addEventListener('click', () => {
    $('question').value = button.dataset.question;
    // 示例问题使用 Demo 默认检索范围，避免沿用上一次的效力筛选导致误拒答。
    $('retrieval-mode').value = 'hybrid';
    $('document-type').value = 'all';
    $('validity').value = 'all';
    $('question').focus();
  });
});

$('upload-button').addEventListener('click', uploadFile); $('build-button').addEventListener('click', buildIndex); $('chat-form').addEventListener('submit', askQuestion); refreshHealth();
