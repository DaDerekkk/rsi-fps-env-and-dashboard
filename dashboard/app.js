/* Main UI controller for FPS Multi-Account Live Dashboard. */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  let state = null;
  let activeMode = 'fps'; // 'fps' | 'racing'
  let selectedAccount = '';
  let selectedMap = '';
  let selectedOutcome = '';
  let selectedRacingAccount = '';
  let selectedRacingPlace = '';
  let autoTimer = null;
  let refreshing = false;

  function esc(str) {
    return String(str ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function fmtTimeCST(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
  }

  function fmtRelative(iso) {
    if (!iso) return '';
    const diff = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
    if (Number.isNaN(diff)) return '';
    if (diff < 60) return `${Math.max(0, diff)}秒前`;
    if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`;
    return `${Math.floor(diff / 3600)}小时前`;
  }

  function showBanner(msg, isError = false) {
    const b = $('banner');
    if (!msg) { b.hidden = true; return; }
    b.hidden = false;
    b.className = 'banner' + (isError ? ' error' : '');
    b.textContent = msg;
  }

  function mapShort(mapId) {
    if (!mapId) return '待解析';
    return mapId.replace(/-cs2-v\d+$/, '').replace(/-v\d+$/, '');
  }

  function renderAccounts() {
    const grid = $('accountGrid');
    grid.replaceChildren();
    if (!state || !state.accounts) return;

    for (const acc of state.accounts) {
      const card = document.createElement('article');
      card.className = 'acc-card' + (selectedAccount === acc.alias ? ' selected' : '');
      card.addEventListener('click', () => {
        selectedAccount = selectedAccount === acc.alias ? '' : acc.alias;
        renderAccounts();
        renderAccountPills();
        renderMatches();
      });

      const st = acc.standing || {};
      const pub = acc.latest_published;
      const curVer = acc.current_version;
      const rm = acc.round_match;
      const sum = acc.recent_summary || { overall: {}, attack: {}, defend: {} };
      const att = sum.attack || {};
      const def = sum.defend || {};
      const ov = sum.overall || {};

      const roleBadge = acc.role === 'primary'
        ? '<span class="badge badge-primary">主账号</span>'
        : '<span class="badge badge-secondary">副账号</span>';
      const rmState = rm ? (rm.state || rm.round_state || '') : '';
      const liveBadge = (rmState === 'running' || rmState === 'assigned' || rmState === 'queued')
        ? `<span class="badge badge-live">● ${esc(rmState)}</span>` : '';

      const verLabel = curVer && curVer.number != null
        ? `v${curVer.number}`
        : pub && pub.number != null ? `v${pub.number}` : '无版本';
      const shaShort = (curVer?.bundle_sha256 || pub?.bundle_sha256 || '').slice(0, 8);
      const rmVer = rm?.python_version?.number != null ? `本轮绑定 v${rm.python_version.number}` : '';

      card.innerHTML = `
        <div class="acc-head">
          <div>
            <div class="acc-name">${esc(acc.username)}</div>
            <div class="acc-alias">${esc(acc.alias)} · ${esc(acc.agent_id.slice(0, 8))}</div>
          </div>
          <div style="display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end;">
            ${liveBadge}${roleBadge}
          </div>
        </div>
        <div class="acc-metrics">
          <div class="kpi">
            <div class="kpi-label">排名</div>
            <div class="kpi-val">${st.rank != null ? '#' + st.rank : '—'}</div>
            <div class="kpi-sub">${st.provisional ? '暂定' : '正式'}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Rating</div>
            <div class="kpi-val">${st.rating != null ? Number(st.rating).toFixed(1) : '—'}</div>
            <div class="kpi-sub">${st.games != null ? st.games + ' 场' : '0 场'}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">胜率</div>
            <div class="kpi-val">${st.win_rate_percent != null ? st.win_rate_percent + '%' : '—'}</div>
            <div class="kpi-sub">近期 ${ov.wins || 0}W-${ov.losses || 0}L</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">最新版本</div>
            <div class="kpi-val">${esc(verLabel)}</div>
            <div class="kpi-sub">${shaShort || '—'}</div>
          </div>
        </div>
        <div class="phase-strip">
          <div class="phase-box">
            <div class="phase-title">进攻方 (T) · ${att.rounds || 0} 回合</div>
            <div class="phase-nums">${att.k || 0}K/${att.d || 0}D/${att.a || 0}A · ADR ${att.adr ?? 0}</div>
            <div class="kpi-sub">胜 ${att.wins || 0}/${att.rounds || 0} · 爆头 ${att.hs_pct ?? 0}% · 孤死 ${att.untraded || 0}</div>
          </div>
          <div class="phase-box ct">
            <div class="phase-title">防守方 (CT) · ${def.rounds || 0} 回合</div>
            <div class="phase-nums">${def.k || 0}K/${def.d || 0}D/${def.a || 0}A · ADR ${def.adr ?? 0}</div>
            <div class="kpi-sub">胜 ${def.wins || 0}/${def.rounds || 0} · 爆头 ${def.hs_pct ?? 0}% · 孤死 ${def.untraded || 0}</div>
          </div>
        </div>
        <div class="acc-foot">
          <span>${rm ? `最近轮次: ${esc(rmState)} ${rmVer}` : '暂无排期对局'} · 异常 ${ov.faults || 0}</span>
          <span>下轮: ${acc.next_round_start_at ? fmtTimeCST(acc.next_round_start_at) : '—'}</span>
        </div>
        ${acc.errors && acc.errors.length ? `<div class="muted" style="color:#fb7185;">⚠️ ${esc(acc.errors.join(' | '))}</div>` : ''}
      `;
      grid.appendChild(card);
    }
  }

  function renderAccountPills() {
    const wrap = $('accountPills');
    wrap.replaceChildren();
    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = 'pill' + (!selectedAccount ? ' active' : '');
    allBtn.textContent = `全部账号 (${state?.matches?.length || 0})`;
    allBtn.addEventListener('click', () => {
      selectedAccount = '';
      renderAccounts(); renderAccountPills(); renderMatches();
    });
    wrap.appendChild(allBtn);

    for (const acc of state?.accounts || []) {
      const cnt = (state.matches || []).filter(m => m.account_alias === acc.alias).length;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pill' + (selectedAccount === acc.alias ? ' active' : '');
      btn.textContent = `${acc.short_name} (${cnt})`;
      btn.addEventListener('click', () => {
        selectedAccount = acc.alias;
        renderAccounts(); renderAccountPills(); renderMatches();
      });
      wrap.appendChild(btn);
    }
  }

  function populateMapFilter() {
    const sel = $('mapFilter');
    const prev = sel.value;
    const maps = [...new Set((state?.matches || []).map(m => m.map_id).filter(Boolean))].sort();
    sel.replaceChildren();
    const opt0 = document.createElement('option');
    opt0.value = ''; opt0.textContent = '全部地图';
    sel.appendChild(opt0);
    for (const m of maps) {
      const o = document.createElement('option');
      o.value = m; o.textContent = mapShort(m);
      if (m === prev) o.selected = true;
      sel.appendChild(o);
    }
  }

  function rankBadge(rank) {
    if (rank == null) return '<span class="muted">—</span>';
    const cls = rank <= 20 ? 'rank-chip top20' : rank <= 50 ? 'rank-chip top50' : 'rank-chip';
    return `<span class="${cls}">#${rank}</span>`;
  }

  function renderMatches() {
    const tbody = $('matchTbody');
    tbody.replaceChildren();
    const rows = (state?.matches || []).filter(m => {
      if (selectedAccount && m.account_alias !== selectedAccount) return false;
      if (selectedMap && m.map_id !== selectedMap) return false;
      if (selectedOutcome && m.outcome !== selectedOutcome) return false;
      return true;
    });

    if (!rows.length) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="9" class="muted" style="text-align:center;padding:28px;">当前筛选条件下暂无对局记录</td>';
      tbody.appendChild(tr);
      return;
    }

    for (const m of rows) {
      const tr = document.createElement('tr');
      const ov = m.own_stats?.overall;
      const att = m.own_stats?.attack;
      const def = m.own_stats?.defend;
      const lb = m.lobby_summary || {};

      const outBadge = m.outcome === 'W' ? '<span class="badge badge-win">胜 W</span>'
        : m.outcome === 'L' ? '<span class="badge badge-loss">负 L</span>'
        : m.outcome === 'D' ? '<span class="badge badge-draw">平 D</span>'
        : `<span class="badge badge-live">${esc(m.status || '进行中')}</span>`;

      const scoreStr = (m.own_score != null && m.opp_score != null) ? `${m.own_score} : ${m.opp_score}` : '—';
      const rc = m.rating_change;
      const deltaHtml = rc && rc.delta != null
        ? `<span class="${rc.delta >= 0 ? 'delta-pos' : 'delta-neg'}">${rc.delta >= 0 ? '+' : ''}${Number(rc.delta).toFixed(2)}</span> <span class="muted">(${Number(rc.after).toFixed(1)})</span>`
        : '<span class="muted">—</span>';

      const bv = m.bound_version;
      let verTag = `<span class="muted">${esc(m.job_id.slice(0, 8))}</span>`;
      if (bv && bv.number != null) {
        const isExact = bv.source && bv.source !== 'inferred_from_bundle_time';
        verTag = isExact
          ? `<span title="平台/回执精确绑定版本">v${bv.number}</span>`
          : `<span class="muted" title="按开赛时间推定（当时最新过审版本），非平台直接回传绑定">~v${bv.number}</span>`;
      }

      const lobbyCell = (lb.ally_avg_rating != null || lb.enemy_avg_rating != null)
        ? `<div class="stat-main">${lb.ally_avg_rating != null ? Math.round(lb.ally_avg_rating) : '—'} <span class="muted">vs</span> ${lb.enemy_avg_rating != null ? Math.round(lb.enemy_avg_rating) : '—'}</div>
           <div class="stat-sub">当时最高 ${lb.ally_best_rating != null ? Math.round(lb.ally_best_rating) : '—'} vs ${lb.enemy_best_rating != null ? Math.round(lb.enemy_best_rating) : '—'}</div>`
        : '<span class="muted">—</span>';

      const overallCell = ov
        ? `<div class="stat-main">${ov.k} / ${ov.d} / ${ov.a} <span class="muted">(${ov.kd})</span></div>
           <div class="stat-sub">ADR ${ov.adr} · 命中 ${ov.acc_pct}% · 爆头 ${ov.hs_pct}% · KAST ${ov.kast_pct}%</div>`
        : '<span class="muted">未加载 / 进行中</span>';

      const attCell = att && att.rounds > 0
        ? `<div class="stat-main" style="color:var(--t-color)">${att.k}/${att.d}/${att.a} · ADR ${att.adr}</div>
           <div class="stat-sub">${att.wins}W-${att.losses}L (${att.rounds}r) · 孤死 ${att.untraded_deaths}</div>`
        : '<span class="muted">—</span>';

      const defCell = def && def.rounds > 0
        ? `<div class="stat-main" style="color:var(--ct-color)">${def.k}/${def.d}/${def.a} · ADR ${def.adr}</div>
           <div class="stat-sub">${def.wins}W-${def.losses}L (${def.rounds}r) · 孤死 ${def.untraded_deaths}</div>`
        : '<span class="muted">—</span>';

      const objFaultCell = ov
        ? `<div class="stat-sub">安包 ${ov.plants} · 拆包 ${ov.defuses}</div>
           <div style="margin-top:3px;">${ov.faults > 0 ? `<span class="badge badge-fault">⚠️ ${ov.faults} Fault</span>` : '<span class="muted">0 异常</span>'}</div>`
        : '<span class="muted">—</span>';

      const isDone = m.status === 'succeeded';

      tr.innerHTML = `
        <td>
          <div style="font-weight:700;">${esc(m.account_short_name)}</div>
          <div class="muted mono">${fmtTimeCST(m.created_at)} · S${m.own_seat ?? '?'}</div>
        </td>
        <td>
          <div class="stat-main">${esc(mapShort(m.map_id))}</div>
          <div class="stat-sub">${verTag}</div>
        </td>
        <td>
          <div style="display:flex;align-items:center;gap:6px;">${outBadge}<span class="score-big">${scoreStr}</span></div>
          <div style="margin-top:2px;">${deltaHtml}</div>
        </td>
        <td>${lobbyCell}</td>
        <td>${overallCell}</td>
        <td>${attCell}</td>
        <td>${defCell}</td>
        <td>${objFaultCell}</td>
        <td>
          <div class="action-group">
            ${isDone ? `<button class="btn btn-xs btn-detail" type="button">详情</button>` : ''}
            ${isDone ? `<button class="btn btn-xs btn-accent btn-2d" type="button">▶ 2D回放</button>` : ''}
            <a class="btn btn-xs btn-ghost" href="${esc(m.agx_url)}" target="_blank" rel="noopener">AGX 3D ↗</a>
          </div>
        </td>
      `;

      const detailBtn = tr.querySelector('.btn-detail');
      if (detailBtn) detailBtn.addEventListener('click', () => openMatchDetail(m));
      const replayBtn = tr.querySelector('.btn-2d');
      if (replayBtn) {
        replayBtn.addEventListener('click', () => {
          window.FPSReplayTheater.open(m.job_id, {
            ownSeat: m.own_seat,
            mapId: m.map_id,
            accountName: m.account_username,
            participants: m.participants
          });
        });
      }
      tbody.appendChild(tr);
    }
  }

  async function openMatchDetail(m) {
    const modal = $('detailModal');
    modal.hidden = false;
    $('detailTitle').textContent = `对局详情 · ${mapShort(m.map_id)} (${m.own_score ?? '?'} : ${m.opp_score ?? '?'})`;
    $('detailSub').textContent = `${m.account_username} (Seat ${m.own_seat}) · ${fmtTimeCST(m.created_at)} · Job ${m.job_id}`;
    $('detailOpenAGX').href = m.agx_url;
    $('detailOpen2D').textContent = '▶ 打开 2D 战术回放';
    $('detailOpen2D').onclick = () => {
      modal.hidden = true;
      window.FPSReplayTheater.open(m.job_id, { ownSeat: m.own_seat, mapId: m.map_id, accountName: m.account_username, participants: m.participants });
    };
    const body = $('detailBody');
    body.innerHTML = '<div class="muted">正在加载完整 10 人计分板与逐回合数据…</div>';

    try {
      const resp = await fetch(`/api/match/${encodeURIComponent(m.job_id)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const st = data.stats;
      const ownSb = st.scoreboard[String(m.own_seat)] || {};
      const ov = ownSb.overall || {};
      const att = ownSb.attack || {};
      const def = ownSb.defend || {};

      // Round pills
      const roundPillsHtml = (st.rounds || []).map(r => {
        const ownTeam = m.own_team;
        const won = r.winner === ownTeam;
        const side = ownTeam === r.attack_team ? 'T' : 'CT';
        const rs = (r.seats || {})[String(m.own_seat)] || {};
        const flt = rs.faults > 0 ? ' has-fault' : '';
        return `
          <div class="r-pill ${won ? 'win' : 'loss'}${flt}" title="R${r.round_num} ${side} | ${r.reason} | ${rs.k}K/${rs.d}D ${rs.dmg}DMG">
            <div style="font-weight:700;color:${side === 'T' ? 'var(--t-color)' : 'var(--ct-color)'}">R${r.round_num} ${side}</div>
            <div>${rs.k || 0}K/${rs.d || 0}D</div>
            <div class="muted" style="font-size:10px;">${Math.round(rs.dmg || 0)} dmg</div>
          </div>
        `;
      }).join('');

      // 10-player scoreboard sorted by team then ADR, enriched with live rank/rating/win_rate
      const participants = m.participants || [];
      const lbSum = m.lobby_summary || {};
      const sbRows = Object.values(st.scoreboard || {}).map(row => {
        const pMeta = participants.find(p => p.seat === row.seat) || {};
        return {
          ...row,
          display_name: pMeta.display_name || `Seat ${row.seat}`,
          is_mine: row.seat === m.own_seat,
          rating_change: pMeta.rating_change,
          rank: pMeta.rank,
          rating: pMeta.rating,
          win_rate_percent: pMeta.win_rate_percent
        };
      }).sort((a, b) => (a.team === m.own_team ? 0 : 1) - (b.team === m.own_team ? 0 : 1) || b.overall.adr - a.overall.adr);

      const sbHtml = sbRows.map(r => {
        const o = r.overall;
        const rc = r.rating_change || {};
        const delta = rc.delta != null
          ? `<span class="${rc.delta >= 0 ? 'delta-pos' : 'delta-neg'}">${rc.delta >= 0 ? '+' : ''}${Number(rc.delta).toFixed(1)}</span>`
          : '—';
        const scoreFlow = (rc.before != null && rc.after != null)
          ? `${Math.round(rc.before)}→${Math.round(rc.after)}`
          : (r.rating != null ? `${Math.round(r.rating)}` : '—');
        const wrStr = r.win_rate_percent != null ? `${r.win_rate_percent}%` : '—';
        return `
          <tr class="${r.is_mine ? 'is-mine' : ''}">
            <td>${r.team === m.own_team ? '己方' : '敌方'} · S${r.seat}</td>
            <td style="font-family:inherit;font-weight:${r.is_mine ? 700 : 500};">${r.is_mine ? '★ ' : ''}${esc(r.display_name)}</td>
            <td>${rankBadge(r.rank)}</td>
            <td>${scoreFlow} <span class="muted">(${wrStr})</span></td>
            <td>${delta}</td>
            <td>${o.k} / ${o.d} / ${o.a}</td>
            <td>${o.kd}</td>
            <td><strong>${o.adr}</strong></td>
            <td>${o.acc_pct}% (${o.hits}/${o.fires})</td>
            <td>${o.hs_pct}% (${o.hs})</td>
            <td>${o.kast_pct}%</td>
            <td>${o.untraded_deaths}</td>
            <td>${o.plants}/${o.defuses}</td>
            <td>${o.faults > 0 ? `<span style="color:#fb7185">${o.faults}</span>` : '0'}</td>
          </tr>
        `;
      }).join('');

      const lobbySummaryLine = (lbSum.ally_avg_rating != null || lbSum.enemy_avg_rating != null)
        ? `<div class="muted" style="margin-bottom:8px;">
             己方当时均分 <strong>${lbSum.ally_avg_rating ?? '—'}</strong>（当时最高 <strong>${lbSum.ally_best_rating ?? '—'}</strong>）
             &nbsp;vs&nbsp;
             敌方当时均分 <strong>${lbSum.enemy_avg_rating ?? '—'}</strong>（当时最高 <strong>${lbSum.enemy_best_rating ?? '—'}</strong>）
             &nbsp;｜&nbsp; 现榜最高：己 #${lbSum.ally_best_rank ?? '—'} vs 敌 #${lbSum.enemy_best_rank ?? '—'}
           </div>`
        : '';

      const faultBox = (st.faults || []).length
        ? `<div class="detail-card"><h4 style="color:#fb7185;">⚠️ 本局运行异常记录 (${st.faults.length})</h4>
           <div class="mono muted">${st.faults.map(f => `R${f.round} tick=${f.tick} Seat=${f.seat} (${esc(f.code)})`).join(' · ')}</div></div>`
        : '';

      body.innerHTML = `
        <div class="detail-phase-grid">
          <div class="detail-card">
            <h4>综合表现 (Seat ${m.own_seat})</h4>
            <div class="stat-main">${ov.k || 0}K / ${ov.d || 0}D / ${ov.a || 0}A · K/D ${ov.kd ?? 0} · ADR ${ov.adr ?? 0}</div>
            <div class="stat-sub">命中 ${ov.hits || 0}/${ov.fires || 0} (${ov.acc_pct ?? 0}%) · 爆头 ${ov.hs || 0} (${ov.hs_pct ?? 0}%) · KAST ${ov.kast_pct ?? 0}%</div>
          </div>
          <div class="detail-card" style="border-left:3px solid var(--t-color)">
            <h4 style="color:var(--t-color)">进攻方 (T) · ${att.rounds || 0} 回合 (${att.wins || 0}W-${att.losses || 0}L)</h4>
            <div class="stat-main">${att.k || 0}K / ${att.d || 0}D / ${att.a || 0}A · ADR ${att.adr ?? 0}</div>
            <div class="stat-sub">命中 ${att.acc_pct ?? 0}% · 爆头 ${att.hs_pct ?? 0}% · 安包 ${att.plants || 0} · 孤死 ${att.untraded_deaths || 0}</div>
          </div>
          <div class="detail-card" style="border-left:3px solid var(--ct-color)">
            <h4 style="color:var(--ct-color)">防守方 (CT) · ${def.rounds || 0} 回合 (${def.wins || 0}W-${def.losses || 0}L)</h4>
            <div class="stat-main">${def.k || 0}K / ${def.d || 0}D / ${def.a || 0}A · ADR ${def.adr ?? 0}</div>
            <div class="stat-sub">命中 ${def.acc_pct ?? 0}% · 爆头 ${def.hs_pct ?? 0}% · 拆包 ${def.defuses || 0} · 孤死 ${def.untraded_deaths || 0}</div>
          </div>
        </div>
        <div class="detail-card">
          <h4>逐回合时间线（${st.total_rounds} 回合）</h4>
          <div class="round-pills">${roundPillsHtml}</div>
        </div>
        ${faultBox}
        <div class="detail-card" style="overflow-x:auto;">
          <h4>全场 10 人详细计分板（含实时排名与分数）</h4>
          ${lobbySummaryLine}
          <table class="sb-table">
            <thead>
              <tr>
                <th>阵营/席位</th><th>玩家</th><th>当前排名</th><th>本场分数 (胜率)</th><th>ΔRating</th><th>K/D/A</th><th>K/D</th><th>ADR</th>
                <th>命中率</th><th>爆头率</th><th>KAST</th><th>孤死</th><th>安/拆</th><th>异常</th>
              </tr>
            </thead>
            <tbody>${sbHtml}</tbody>
          </table>
        </div>
      `;
    } catch (err) {
      body.innerHTML = `<div class="banner error">无法加载对局详情：${esc(err.message)}</div>`;
    }
  }

  function switchMode(mode) {
    activeMode = mode === 'racing' ? 'racing' : 'fps';
    const isFps = activeMode === 'fps';
    $('fpsView').hidden = !isFps;
    $('racingView').hidden = isFps;
    $('tabFps').classList.toggle('active', isFps);
    $('tabFps').setAttribute('aria-selected', String(isFps));
    $('tabRacing').classList.toggle('active', !isFps);
    $('tabRacing').setAttribute('aria-selected', String(!isFps));
    updateStatusLine();
  }

  function updateStatusLine() {
    if (!state) return;
    const count = state.accounts?.length || 0;
    const mCount = state.matches?.length || 0;
    const rCount = state.racing_accounts?.length || 0;
    const rmCount = state.racing_matches?.length || 0;
    if ($('tabFpsCount')) $('tabFpsCount').textContent = `(${count})`;
    if ($('tabRacingCount')) $('tabRacingCount').textContent = `(${rCount})`;
    $('statusLine').textContent = activeMode === 'fps'
      ? `FPS · 已接入 ${count} 个账号 · 共 ${mCount} 场近期对局 · 最近刷新：${fmtTimeCST(state.refreshed_at)} (${fmtRelative(state.refreshed_at)})`
      : `赛车 · 已识别 ${rCount} 个账号 · 共 ${rmCount} 场近期对局 · 最近刷新：${fmtTimeCST(state.refreshed_at)} (${fmtRelative(state.refreshed_at)})`;
    if (activeMode === 'fps' && state.leaderboard_url) {
      $('leaderboardLink').href = state.leaderboard_url;
      $('leaderboardLink').textContent = 'AGX FPS 排行榜 ↗';
    } else if (activeMode === 'racing' && state.racing_leaderboard_url) {
      $('leaderboardLink').href = state.racing_leaderboard_url;
      $('leaderboardLink').textContent = 'AGX 赛车排行榜 ↗';
    }
  }

  const RACING_COLORS = ['#38bdf8', '#facc15', '#4ade80', '#fb923c', '#a78bfa'];

  function renderRacingAccountPills() {
    const wrap = $('racingAccountPills');
    if (!wrap) return;
    wrap.replaceChildren();
    const rMatches = state?.racing_matches || [];
    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = 'pill' + (!selectedRacingAccount ? ' active' : '');
    allBtn.textContent = `全部账号 (${rMatches.length})`;
    allBtn.addEventListener('click', () => {
      selectedRacingAccount = '';
      renderRacing();
    });
    wrap.appendChild(allBtn);

    for (const acc of state?.racing_accounts || []) {
      const cnt = rMatches.filter(m => m.account_alias === acc.alias).length;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pill' + (selectedRacingAccount === acc.alias ? ' active' : '');
      btn.textContent = `${acc.short_name} (${cnt})`;
      btn.addEventListener('click', () => {
        selectedRacingAccount = selectedRacingAccount === acc.alias ? '' : acc.alias;
        renderRacing();
      });
      wrap.appendChild(btn);
    }
  }

  function renderRacingCharts() {
    const bar = $('racingGlobalBar');
    const wrap = $('racingChartWrap');
    const strip = $('racingDeltaStrip');
    if (!bar || !wrap || !strip) return;

    const rAccs = state?.racing_accounts || [];
    const gSum = state?.racing_summary || {};
    const focusAcc = selectedRacingAccount ? rAccs.find(a => a.alias === selectedRacingAccount) : null;

    if (focusAcc) {
      const st = focusAcc.standing || {};
      const rs = focusAcc.rating_stats || {};
      const rp = st.race_performance || {};
      const net = rs.net_from_1000;
      const netCls = net != null ? (net >= 0 ? 'delta-pos' : 'delta-neg') : '';
      const lastD = rs.latest_delta;
      const lastCls = lastD != null ? (lastD >= 0 ? 'delta-pos' : 'delta-neg') : '';
      bar.innerHTML = `
        <div class="gkpi"><div class="gkpi-label">当前聚焦账号</div><div class="gkpi-val" style="font-size:14px;">${esc(focusAcc.username)}</div></div>
        <div class="gkpi"><div class="gkpi-label">当前排名 / 分数</div><div class="gkpi-val">#${st.rank ?? '—'} · ${st.rating != null ? Number(st.rating).toFixed(1) : '—'}</div></div>
        <div class="gkpi"><div class="gkpi-label">累计净涨分 (相对1000)</div><div class="gkpi-val ${netCls}">${net != null ? (net >= 0 ? '+' : '') + net.toFixed(2) : '—'}</div></div>
        <div class="gkpi"><div class="gkpi-label">最近一场 ΔRating</div><div class="gkpi-val ${lastCls}">${lastD != null ? (lastD >= 0 ? '+' : '') + lastD.toFixed(2) : '—'}</div></div>
        <div class="gkpi"><div class="gkpi-label">完赛 / 平均名次 / 前三</div><div class="gkpi-val">${st.games ?? 0}场 · P${rp.average_position != null ? Number(rp.average_position).toFixed(1) : '—'} · ${rp.top_three_percent != null ? Math.round(rp.top_three_percent) + '%' : '—'}</div></div>
      `;
    } else {
      const totGain = gSum.total_net_gain;
      const gainCls = totGain != null ? (totGain >= 0 ? 'delta-pos' : 'delta-neg') : '';
      bar.innerHTML = `
        <div class="gkpi"><div class="gkpi-label">全局参赛账号</div><div class="gkpi-val">${gSum.active_accounts ?? 0} 活跃 / ${gSum.total_accounts ?? rAccs.length} 已报名</div></div>
        <div class="gkpi"><div class="gkpi-label">全局最高排名</div><div class="gkpi-val">${gSum.best_rank != null ? '#' + gSum.best_rank : '—'}</div></div>
        <div class="gkpi"><div class="gkpi-label">全局最高 Rating</div><div class="gkpi-val">${gSum.best_rating != null ? gSum.best_rating.toFixed(1) : '—'}</div></div>
        <div class="gkpi"><div class="gkpi-label">活跃号平均 Rating</div><div class="gkpi-val">${gSum.avg_rating != null ? gSum.avg_rating.toFixed(1) : '—'}</div></div>
        <div class="gkpi"><div class="gkpi-label">全局累计净涨分</div><div class="gkpi-val ${gainCls}">${totGain != null ? (totGain >= 0 ? '+' : '') + totGain.toFixed(1) : '—'}</div></div>
      `;
    }

    // Build SVG curves from score_history (falling back to recent_matches)
    const series = rAccs.map((acc, idx) => {
      const sh = acc.score_history || [];
      const matchesById = Object.fromEntries((acc.recent_matches || []).map(m => [m.job_id, m]));
      const pts = [{ label: '初始', rating: 1000, delta: 0, place: null, ver: null, t: null }];
      if (sh.length) {
        let prev = 1000;
        for (let i = 0; i < sh.length; i++) {
          const item = sh[i];
          const mm = matchesById[item.job_id];
          const delta = mm?.rating_change?.delta != null ? mm.rating_change.delta : (item.rating - prev);
          prev = item.rating;
          pts.push({
            label: `#${i + 1}`,
            rating: item.rating,
            delta,
            place: mm?.place ?? null,
            ver: item.version_number != null ? `v${item.version_number}` : '',
            t: item.t,
          });
        }
      }
      return { acc, color: RACING_COLORS[idx % RACING_COLORS.length], pts };
    }).filter(s => s.pts.length > 1);

    if (!series.length) {
      wrap.innerHTML = '<div class="muted" style="padding:18px;text-align:center;">暂无足够的完赛记录绘制分数曲线</div>';
      strip.replaceChildren();
      return;
    }

    const allRatings = series.flatMap(s => s.pts.map(p => p.rating));
    const minY = Math.floor((Math.min(1000, ...allRatings) - 40) / 50) * 50;
    const maxY = Math.ceil((Math.max(1100, ...allRatings) + 40) / 50) * 50;
    const maxX = Math.max(...series.map(s => s.pts.length - 1), 1);

    const W = 860, H = 200, padL = 52, padR = 20, padT = 16, padB = 26;
    const xPos = i => padL + (i / maxX) * (W - padL - padR);
    const yPos = r => padT + (1 - (r - minY) / Math.max(1, maxY - minY)) * (H - padT - padB);

    let gridSvg = '';
    const steps = 4;
    for (let k = 0; k <= steps; k++) {
      const val = minY + ((maxY - minY) * k) / steps;
      const y = yPos(val).toFixed(1);
      gridSvg += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="rgba(148,163,184,0.14)" stroke-width="1"/>`;
      gridSvg += `<text x="${padL - 8}" y="${Number(y) + 4}" text-anchor="end" fill="#8ea1b8" font-size="10" font-family="ui-monospace,monospace">${Math.round(val)}</text>`;
    }
    const base1000Y = yPos(1000).toFixed(1);
    gridSvg += `<line x1="${padL}" y1="${base1000Y}" x2="${W - padR}" y2="${base1000Y}" stroke="rgba(250,204,21,0.35)" stroke-dasharray="4,4" stroke-width="1"/>`;

    let linesSvg = '';
    for (const s of series) {
      const isFocused = !selectedRacingAccount || selectedRacingAccount === s.acc.alias;
      const opacity = isFocused ? '1' : '0.22';
      const sw = isFocused ? '2.5' : '1.5';
      const polyPts = s.pts.map((p, i) => `${xPos(i).toFixed(1)},${yPos(p.rating).toFixed(1)}`).join(' ');
      linesSvg += `<polyline fill="none" stroke="${s.color}" stroke-width="${sw}" stroke-opacity="${opacity}" points="${polyPts}"/>`;
      for (let i = 0; i < s.pts.length; i++) {
        const p = s.pts[i];
        const cx = xPos(i).toFixed(1), cy = yPos(p.rating).toFixed(1);
        const tip = `${s.acc.short_name} ${p.label}: ${p.rating.toFixed(1)} (${p.delta >= 0 ? '+' : ''}${p.delta.toFixed(1)})${p.place ? ' · P' + p.place : ''}${p.ver ? ' · ' + p.ver : ''}`;
        linesSvg += `<circle cx="${cx}" cy="${cy}" r="${isFocused ? 4 : 2.5}" fill="${s.color}" fill-opacity="${opacity}" stroke="#0b1017" stroke-width="1"><title>${esc(tip)}</title></circle>`;
      }
    }
    let xAxisSvg = '';
    for (let i = 0; i <= maxX; i++) {
      xAxisSvg += `<text x="${xPos(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" fill="#8ea1b8" font-size="10" font-family="ui-monospace,monospace">${i === 0 ? '初始' : 'G' + i}</text>`;
    }

    const legendHtml = series.map(s => {
      const dim = selectedRacingAccount && selectedRacingAccount !== s.acc.alias ? ' dim' : '';
      const last = s.pts[s.pts.length - 1];
      const net = last.rating - 1000;
      return `<span class="legend-item${dim}" data-alias="${esc(s.acc.alias)}"><i class="legend-swatch" style="background:${s.color}"></i><strong>${esc(s.acc.short_name)}</strong> <span class="mono">${last.rating.toFixed(1)} (${net >= 0 ? '+' : ''}${net.toFixed(1)})</span></span>`;
    }).join('');

    wrap.innerHTML = `
      <div class="chart-legend">${legendHtml}</div>
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;max-height:220px;display:block;">${gridSvg}${xAxisSvg}${linesSvg}</svg>
    `;
    wrap.querySelectorAll('.legend-item').forEach(el => {
      el.addEventListener('click', () => {
        const al = el.getAttribute('data-alias');
        selectedRacingAccount = selectedRacingAccount === al ? '' : al;
        renderRacing();
      });
    });

    // Per-match delta chips for focused account (or latest 8 across all accounts)
    strip.replaceChildren();
    const chipSource = focusAcc
      ? [...(focusAcc.recent_matches || [])].sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''))
      : [...(state?.racing_matches || [])].slice(0, 8).reverse();
    for (const m of chipSource) {
      const rc = m.rating_change;
      if (!rc || rc.delta == null) continue;
      const chip = document.createElement('div');
      chip.className = 'delta-chip';
      const dCls = rc.delta >= 0 ? 'delta-pos' : 'delta-neg';
      const ver = m.bound_version?.number != null ? `v${m.bound_version.number}` : '';
      chip.innerHTML = `
        <div class="muted">${esc(m.account_short_name)} · ${fmtTimeCST(m.created_at)} ${ver ? '· ' + ver : ''}</div>
        <div><strong class="${dCls}">${rc.delta >= 0 ? '+' : ''}${Number(rc.delta).toFixed(1)}</strong> · ${Math.round(rc.before)}→${Math.round(rc.after)} · ${m.place ? 'P' + m.place : '—'}</div>
      `;
      strip.appendChild(chip);
    }
  }

  function openRacingMatchDetail(m) {
    const modal = $('detailModal');
    modal.hidden = false;
    const bv = m.bound_version;
    const verStr = bv && bv.number != null ? `v${bv.number}` : '—';
    $('detailTitle').textContent = `赛车对局详情 · 第 ${m.place ?? '?'} / ${m.total_racers || 10} 名 (${verStr})`;
    $('detailSub').textContent = `${m.account_username} (Seat ${m.own_seat}) · ${fmtTimeCST(m.created_at)} · Job ${m.job_id}`;
    $('detailOpenAGX').href = m.agx_url;
    $('detailOpen2D').textContent = '▶ 打开 2D 赛道回放';
    $('detailOpen2D').onclick = () => {
      modal.hidden = true;
      window.RacingReplayTheater.open(m.job_id, { ownSeat: m.own_seat, accountName: m.account_username });
    };

    const ps = [...(m.participants || [])].sort((a, b) => (a.place ?? 99) - (b.place ?? 99));
    const rowsHtml = ps.map(p => {
      const rc = p.rating_change || {};
      const delta = rc.delta != null
        ? `<span class="${rc.delta >= 0 ? 'delta-pos' : 'delta-neg'}">${rc.delta >= 0 ? '+' : ''}${Number(rc.delta).toFixed(2)}</span>`
        : '—';
      const scoreFlow = (rc.before != null && rc.after != null)
        ? `${Number(rc.before).toFixed(1)} → ${Number(rc.after).toFixed(1)}`
        : (p.rating != null ? `${Number(p.rating).toFixed(1)}` : '—');
      return `
        <tr class="${p.is_mine ? 'is-mine' : ''}">
          <td><strong>P${p.place ?? '?'}</strong></td>
          <td>Seat ${p.seat}</td>
          <td style="font-family:inherit;font-weight:${p.is_mine ? 700 : 500};">${p.is_mine ? '★ ' : ''}${esc(p.display_name)}</td>
          <td>${rankBadge(p.rank)}</td>
          <td>${scoreFlow}</td>
          <td>${delta}</td>
          <td>${p.win_rate_percent != null ? p.win_rate_percent + '%' : '—'}</td>
        </tr>
      `;
    }).join('');

    const lb = m.lobby_summary || {};
    $('detailBody').innerHTML = `
      <div class="detail-card" style="overflow-x:auto;">
        <h4>全场 10 车完赛名次与评分变化</h4>
        <div class="muted" style="margin-bottom:8px;">
          当时大厅均分 <strong>${lb.avg_rating ?? '—'}</strong> · 当时最高分 <strong>${lb.best_rating ?? '—'}</strong>
        </div>
        <table class="sb-table">
          <thead>
            <tr><th>完赛名次</th><th>席位</th><th>车手</th><th>当前排名</th><th>本场分数 (赛前→赛后)</th><th>ΔRating</th><th>现榜胜率</th></tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>
    `;
  }

  function renderRacing() {
    const grid = $('racingAccountGrid');
    const cardWrap = $('racingTableCard');
    const tbody = $('racingMatchTbody');
    if (!grid || !tbody) return;
    grid.replaceChildren();
    tbody.replaceChildren();

    if (state?.racing_leaderboard_url && $('racingLeaderboardLink')) {
      $('racingLeaderboardLink').href = state.racing_leaderboard_url;
    }

    const rAccs = state?.racing_accounts || [];
    if (!rAccs.length) {
      grid.innerHTML = '<div class="muted">当前已接入账号中暂无赛车参赛记录。</div>';
      if (cardWrap) cardWrap.hidden = true;
      return;
    }

    for (const acc of rAccs) {
      const card = document.createElement('article');
      card.className = 'acc-card' + (selectedRacingAccount === acc.alias ? ' selected' : '');
      card.addEventListener('click', () => {
        selectedRacingAccount = selectedRacingAccount === acc.alias ? '' : acc.alias;
        renderRacing();
      });
      const st = acc.standing || {};
      const rp = st.race_performance || {};
      const rs = acc.rating_stats || {};
      const pub = acc.latest_published;
      const rm = acc.round_match;
      const rmState = rm ? (rm.state || rm.round_state || '') : '';
      const verLabel = pub && pub.number != null ? `v${pub.number}` : '无版本';
      const avgPos = rp.average_position != null ? `P${Number(rp.average_position).toFixed(1)}` : '—';
      const top3 = rp.top_three_percent != null ? `${Number(rp.top_three_percent).toFixed(0)}%` : '—';
      const netStr = rs.net_from_1000 != null ? `${rs.net_from_1000 >= 0 ? '+' : ''}${rs.net_from_1000.toFixed(1)}` : '—';
      const lastD = rs.latest_delta != null ? `上场 ${rs.latest_delta >= 0 ? '+' : ''}${rs.latest_delta.toFixed(1)}` : `总净变化 ${netStr}`;
      card.innerHTML = `
        <div class="acc-head">
          <div>
            <div class="acc-name">🏎️ ${esc(acc.username)}</div>
            <div class="acc-alias">${esc(acc.alias)} · ${(acc.agent_id || '').slice(0, 8)}</div>
          </div>
          <span class="badge badge-secondary">赛车 · ${esc(verLabel)}</span>
        </div>
        <div class="acc-metrics">
          <div class="kpi">
            <div class="kpi-label">赛车排名</div>
            <div class="kpi-val">${st.rank != null ? '#' + st.rank : '—'}</div>
            <div class="kpi-sub">${st.provisional ? '暂定' : '正式'} · ${st.games ?? 0} 场</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Rating (${netStr})</div>
            <div class="kpi-val">${st.rating != null ? Number(st.rating).toFixed(1) : '—'}</div>
            <div class="kpi-sub">${esc(lastD)}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">平均名次</div>
            <div class="kpi-val">${avgPos}</div>
            <div class="kpi-sub">完赛 ${rp.classified_matches ?? st.games ?? 0} 场</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">前三率</div>
            <div class="kpi-val">${top3}</div>
            <div class="kpi-sub">${verLabel}</div>
          </div>
        </div>
        <div class="acc-foot">
          <span>${rm ? `最近轮次: ${esc(rmState)}` : '暂无排期对局'}</span>
          <span>下轮: ${acc.next_round_start_at ? fmtTimeCST(acc.next_round_start_at) : '—'}</span>
        </div>
      `;
      grid.appendChild(card);
    }

    renderRacingAccountPills();
    renderRacingCharts();

    const rMatches = (state?.racing_matches || []).filter(m => {
      if (selectedRacingAccount && m.account_alias !== selectedRacingAccount) return false;
      if (selectedRacingPlace) {
        const cap = Number(selectedRacingPlace);
        if (m.place == null || m.place > cap) return false;
      }
      return true;
    });
    if (!rMatches.length) {
      if (cardWrap) cardWrap.hidden = false;
      tbody.innerHTML = '<tr><td colspan="6" class="muted" style="text-align:center;padding:24px;">当前筛选条件下暂无赛车对局记录</td></tr>';
      return;
    }
    if (cardWrap) cardWrap.hidden = false;
    for (const m of rMatches) {
      const tr = document.createElement('tr');
      const bv = m.bound_version;
      const verTag = bv && bv.number != null
        ? (bv.source !== 'inferred_from_bundle_time' ? `v${bv.number}` : `~v${bv.number}`)
        : m.job_id.slice(0, 8);
      const rc = m.rating_change;
      const deltaHtml = rc && rc.delta != null
        ? `<div class="stat-main"><span class="${rc.delta >= 0 ? 'delta-pos' : 'delta-neg'}">${rc.delta >= 0 ? '+' : ''}${Number(rc.delta).toFixed(2)}</span></div>
           <div class="stat-sub">${Number(rc.before).toFixed(1)} → ${Number(rc.after).toFixed(1)}</div>`
        : '<span class="muted">—</span>';
      const placeHtml = m.place != null
        ? `<span class="badge ${m.place <= 3 ? 'badge-win' : m.place <= 6 ? 'badge-draw' : 'badge-loss'}">第 ${m.place} / ${m.total_racers || '?'} 名</span>`
        : `<span class="badge badge-live">${esc(m.status || '进行中')}</span>`;
      const lb = m.lobby_summary || {};
      const lobbyCell = (lb.avg_rating != null || lb.best_rating != null)
        ? `<div class="stat-main">均 ${Math.round(lb.avg_rating)} · 最高 ${Math.round(lb.best_rating)}</div>
           <div class="stat-sub">${(m.participants || []).length} 辆同场</div>`
        : `<span class="muted">${(m.participants || []).length} 人同场</span>`;
      const isDone = m.status === 'succeeded';
      tr.innerHTML = `
        <td><div style="font-weight:700;">${esc(m.account_short_name)}</div><div class="muted mono">${fmtTimeCST(m.created_at)} · S${m.own_seat ?? '?'}</div></td>
        <td><div class="stat-main">${esc(verTag)}</div><div class="stat-sub">${esc(m.job_id.slice(0, 8))}</div></td>
        <td>${placeHtml}</td>
        <td>${deltaHtml}</td>
        <td>${lobbyCell}</td>
        <td>
          <div class="action-group">
            ${isDone ? `<button class="btn btn-xs btn-detail" type="button">详情</button>` : ''}
            ${isDone ? `<button class="btn btn-xs btn-accent btn-race-2d" type="button">▶ 2D回放</button>` : ''}
            <a class="btn btn-xs btn-ghost" href="${esc(m.agx_url)}" target="_blank" rel="noopener">AGX 3D ↗</a>
          </div>
        </td>
      `;
      const detBtn = tr.querySelector('.btn-detail');
      if (detBtn) detBtn.addEventListener('click', () => openRacingMatchDetail(m));
      const r2d = tr.querySelector('.btn-race-2d');
      if (r2d) {
        r2d.addEventListener('click', () => {
          window.RacingReplayTheater.open(m.job_id, {
            ownSeat: m.own_seat,
            accountName: m.account_username
          });
        });
      }
      tbody.appendChild(tr);
    }
  }

  function renderAll() {
    updateStatusLine();
    renderAccounts();
    renderAccountPills();
    populateMapFilter();
    renderMatches();
    renderRacing();
  }

  async function loadInitial() {
    try {
      const resp = await fetch('/api/state');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state = await resp.json();
      showBanner('');
      renderAll();
    } catch (err) {
      showBanner(`初始化加载失败：${err.message}，请点击右上角「一键实时刷新」重试。`, true);
    }
  }

  async function triggerLiveRefresh() {
    if (refreshing) return;
    refreshing = true;
    const btn = $('refreshBtn');
    const icon = $('refreshIcon');
    const label = $('refreshLabel');
    btn.disabled = true;
    icon.classList.add('spinning');
    label.textContent = '正在实时拉取…';
    try {
      const resp = await fetch('/api/refresh', { method: 'POST' });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      state = await resp.json();
      showBanner('');
      renderAll();
    } catch (err) {
      showBanner(`实时刷新失败：${err.message}`, true);
    } finally {
      refreshing = false;
      btn.disabled = false;
      icon.classList.remove('spinning');
      label.textContent = '一键实时刷新';
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    $('tabFps').addEventListener('click', () => switchMode('fps'));
    $('tabRacing').addEventListener('click', () => switchMode('racing'));
    $('refreshBtn').addEventListener('click', triggerLiveRefresh);
    $('mapFilter').addEventListener('change', e => { selectedMap = e.target.value; renderMatches(); });
    $('outcomeFilter').addEventListener('change', e => { selectedOutcome = e.target.value; renderMatches(); });
    if ($('racingPlaceFilter')) {
      $('racingPlaceFilter').addEventListener('change', e => { selectedRacingPlace = e.target.value; renderRacing(); });
    }
    $('detailClose').addEventListener('click', () => { $('detailModal').hidden = true; });
    $('detailModal').addEventListener('click', e => { if (e.target === $('detailModal')) $('detailModal').hidden = true; });
    $('autoRefreshToggle').addEventListener('change', e => {
      if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
      if (e.target.checked) autoTimer = setInterval(triggerLiveRefresh, 120000);
    });
    setInterval(updateStatusLine, 15000);
    loadInitial();
  });
})();
