const $ = (id) => document.getElementById(id);

function setText(id, text) { $(id).textContent = text; }

async function refreshHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    setText('health', data.status === 'ok' ? '服务正常' : '服务异常');
    setText('index-state', data.index_ready ? '已建库' : '未建库');
    const provider = data.llm_provider === 'api' ? `API：${data.model_name}` : '本地 GGUF';
    setText('model-state', data.model_configured ? `${provider}｜${data.model_loaded ? '已加载' : '待加载'}` : `${provider}｜未配置`);
  } catch (error) {
    setText('health', '无法连接服务');
    setText('model-state', '请先启动 FastAPI');
  }
}

async function uploadFile() {
  const file = $('file-input').files[0];
  if (!file) { setText('index-message', '请选择一个文档。'); return; }
  const form = new FormData();
  form.append('file', file);
  $('upload-button').disabled = true;
  setText('index-message', '上传中...');
  try {
    const response = await fetch('/api/documents/upload', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '上传失败');
    setText('index-message', `${data.filename} 已上传，请重建索引。`);
  } catch (error) { setText('index-message', error.message); }
  finally { $('upload-button').disabled = false; }
}

async function buildIndex() {
  $('build-button').disabled = true;
  setText('index-message', '正在解析文档并建立索引，首次运行可能需要下载向量模型...');
  try {
    const response = await fetch('/api/index/build', { method: 'POST' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '建库失败');
    setText('index-message', `完成：${data.document_sections} 个文档区段，${data.chunks} 个检索片段。`);
    await refreshHealth();
  } catch (error) { setText('index-message', error.message); }
  finally { $('build-button').disabled = false; }
}

function addMessage(role, text = '') {
  const empty = document.querySelector('.empty-state');
  if (empty) empty.remove();
  const wrapper = document.createElement('article');
  wrapper.className = `message ${role}`;
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? '你' : '本地模型';
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text;
  wrapper.append(label, body);
  $('messages').appendChild(wrapper);
  $('messages').scrollTop = $('messages').scrollHeight;
  return { wrapper, body };
}

function addSources(wrapper, data) {
  const section = document.createElement('div');
  section.className = 'sources';
  const info = document.createElement('p');
  info.className = 'retrieval-info';
  const labels = { dense: '向量检索', bm25: 'BM25', hybrid: '混合检索', none: '未检索' };
  const reranker = data.reranker_backend === 'none' ? '未执行' : data.reranker_backend;
  info.textContent = `检索查询：${data.query}｜方式：${labels[data.mode] || data.mode}｜重排序：${reranker}`;
  section.appendChild(info);
  if (data.sources.length) {
    const title = document.createElement('div');
    title.textContent = '检索来源';
    section.appendChild(title);
  }
  data.sources.forEach((source, index) => {
    const details = document.createElement('details');
    details.className = 'source-item';
    const summary = document.createElement('summary');
    const method = source.methods.length ? `｜${source.methods.join('+')}` : '';
    const score = source.rerank_score === null ? '' : `｜重排 ${source.rerank_score.toFixed(3)}`;
    summary.textContent = `[${index + 1}] ${source.file}${source.page ? `，第 ${source.page} 页` : ''}${method}${score}`;
    const excerpt = document.createElement('p');
    excerpt.className = 'source-excerpt';
    excerpt.textContent = source.excerpt;
    details.append(summary, excerpt);
    section.appendChild(details);
  });
  wrapper.appendChild(section);
}

async function askQuestion(event) {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question) return;
  addMessage('user', question);
  const assistant = addMessage('assistant');
  $('question').value = '';
  $('question').disabled = true;
  setText('chat-state', '模型生成中...');
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        use_rag: $('rag-toggle').checked,
        top_k: Number($('top-k').value),
        retrieval_mode: $('retrieval-mode').value,
        rewrite_query: $('rewrite-toggle').checked,
        rerank: $('rerank-toggle').checked,
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '问答失败');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const eventData = JSON.parse(line);
        if (eventData.event === 'sources') addSources(assistant.wrapper, eventData);
        if (eventData.event === 'token') {
          assistant.body.textContent += eventData.text;
          $('messages').scrollTop = $('messages').scrollHeight;
        }
        if (eventData.event === 'error') throw new Error(eventData.message);
      }
      if (done) break;
    }
    setText('chat-state', '就绪');
  } catch (error) {
    assistant.body.textContent = `请求失败：${error.message}`;
    setText('chat-state', '请求失败');
  } finally {
    $('question').disabled = false;
    $('question').focus();
  }
}

$('upload-button').addEventListener('click', uploadFile);
$('build-button').addEventListener('click', buildIndex);
$('chat-form').addEventListener('submit', askQuestion);
refreshHealth();
