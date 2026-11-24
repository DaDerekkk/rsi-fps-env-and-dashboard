/* Interactive 2D Tactical Replay Theater for AGX FPS public replays. */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const KEY_EVENT_TYPES = new Set([
    'kill', 'fault', 'plant_complete_detail', 'plant_complete',
    'defuse_complete_detail', 'defuse_complete', 'bomb_exploded', 'bomb_defused'
  ]);

  let bundle = null;
  let activeJobId = null;
  let ownSeat = null;
  let extraMetaBySeat = {};
  let roundIdx = 0;
  let curT = 0;
  let playDir = 0; // 0 = paused, 1 = forward, -1 = reverse
  let speed = 1;
  let lastRaf = 0;
  let rafId = null;
  let hudTick = 0;
  let pollTimer = null;
  let readyRoundsSet = new Set();
  let autoStarted = false;

  // Track point tuple: [t, x, y, z, yaw, hp, alive, crouch, weapon]
  function sampleTrack(track, t) {
    if (!track || !track.length) return null;
    let lo = 0, hi = track.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (track[mid][0] <= t) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return track[0];
    const p0 = track[lo - 1];
    const p1 = track[lo];
    if (!p1 || p0[6] === 0 || p1[0] - p0[0] > 0.6) return p0;
    const alpha = Math.max(0, Math.min(1, (t - p0[0]) / Math.max(1e-4, p1[0] - p0[0])));
    // Shortest-angle interpolation for yaw
    let dyaw = p1[4] - p0[4];
    while (dyaw > Math.PI) dyaw -= Math.PI * 2;
    while (dyaw < -Math.PI) dyaw += Math.PI * 2;
    return [
      t,
      p0[1] + (p1[1] - p0[1]) * alpha,
      p0[2] + (p1[2] - p0[2]) * alpha,
      p0[3] + (p1[3] - p0[3]) * alpha,
      p0[4] + dyaw * alpha,
      p0[5],
      p0[6],
      p0[7],
      p0[8]
    ];
  }

  function sampleBomb(bombTrack, t) {
    if (!bombTrack || !bombTrack.length) return null;
    let lo = 0, hi = bombTrack.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (bombTrack[mid][0] <= t) lo = mid + 1;
      else hi = mid;
    }
    return bombTrack[Math.max(0, lo - 1)] || null;
  }

  function makeProjection(bounds, w, h) {
    const pad = 36;
    const dx = Math.max(1, bounds[2] - bounds[0]);
    const dy = Math.max(1, bounds[3] - bounds[1]);
    const scale = Math.min((w - pad * 2) / dx, (h - pad * 2) / dy);
    const ox = (w - dx * scale) / 2;
    const oy = (h - dy * scale) / 2;
    return {
      scale,
      xy: (x, y) => [ox + (x - bounds[0]) * scale, h - oy - (y - bounds[1]) * scale]
    };
  }

  function renderScrubberMarkers(round) {
    const host = $('rpEventMarkers');
    if (!host) return;
    host.replaceChildren();
    const span = Math.max(0.1, round.duration - round.min_t);
    for (const ev of round.events) {
      if (!KEY_EVENT_TYPES.has(ev.type)) continue;
      const pct = Math.max(0, Math.min(100, ((ev.t - round.min_t) / span) * 100));
      const tick = document.createElement('span');
      tick.className = 'scrubber-tick ' + (
        ev.type === 'kill' ? 'kill' : ev.type === 'fault' ? 'fault' : 'obj'
      );
      tick.style.left = `${pct.toFixed(2)}%`;
      host.appendChild(tick);
    }
  }

  function playerLabel(seat) {
    if (!bundle) return `S${seat}`;
    const p = bundle.participants.find(x => x.seat === seat);
    if (!p) return `S${seat}`;
    const name = p.display_name || `S${seat}`;
    return name.length > 10 ? name.slice(0, 9) + '…' : name;
  }

  function seatSide(round, seat) {
    const team = seat < 5 ? 0 : 1;
    return team === round.attack_team ? 'T' : 'CT';
  }

  function renderRoundBar() {
    const bar = $('replayRoundBar');
    bar.replaceChildren();
    const ownTeam = ownSeat != null ? (ownSeat < 5 ? 0 : 1) : 0;
    bundle.rounds.forEach((r, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      const won = r.winner === ownTeam;
      const isReady = readyRoundsSet.size === 0 || readyRoundsSet.has(r.index) || (r.tracks && Object.values(r.tracks).some(tr => tr.length > 0));
      btn.className = 'rb-btn' + (r.winner != null ? (won ? ' win' : ' loss') : '') + (idx === roundIdx ? ' active' : '') + (!isReady ? ' pending' : '');
      const side = ownSeat != null ? seatSide(r, ownSeat) : '';
      btn.textContent = `R${r.round_num}${side ? ' ' + side : ''}`;
      btn.title = isReady ? `第 ${r.round_num} 回合 (${r.reason || '完成'})` : `第 ${r.round_num} 回合（正在后台缓存中…）`;
      btn.addEventListener('click', () => selectRound(idx));
      bar.appendChild(btn);
    });
  }

  function renderEventsList(round) {
    const box = $('rpEventsList');
    box.replaceChildren();
    const keyEvs = round.events.filter(ev => KEY_EVENT_TYPES.has(ev.type));
    if (!keyEvs.length) {
      const empty = document.createElement('div');
      empty.className = 'muted';
      empty.textContent = '本回合无关键击杀/目标事件';
      box.appendChild(empty);
      return;
    }
    keyEvs.forEach(ev => {
      const div = document.createElement('div');
      div.className = 'ev-item ' + (
        ev.type === 'kill' ? 'ev-kill' :
        ev.type === 'fault' ? 'ev-fault' : 'ev-obj'
      );
      let desc = ev.type;
      if (ev.type === 'kill') {
        const hs = ev.part === 'head' ? ' 🎯HS' : '';
        desc = `${playerLabel(ev.killer)} 🔫 ${playerLabel(ev.target)}${hs}`;
      } else if (ev.type.startsWith('plant')) {
        desc = `💣 ${playerLabel(ev.player)} 安包完成${ev.site ? ' · ' + ev.site : ''}`;
      } else if (ev.type.startsWith('defuse')) {
        desc = `🔧 ${playerLabel(ev.player)} 拆包完成`;
      } else if (ev.type === 'bomb_exploded') {
        desc = '💥 炸弹爆炸';
      } else if (ev.type === 'bomb_defused') {
        desc = '✅ 炸弹已拆除';
      } else if (ev.type === 'fault') {
        desc = `⚠️ 异常 FAULT (${playerLabel(ev.player)})`;
      }
      const left = document.createElement('span');
      left.textContent = desc;
      const right = document.createElement('span');
      right.className = 'mono muted';
      right.textContent = `${ev.t.toFixed(1)}s`;
      div.append(left, right);
      div.addEventListener('click', () => {
        curT = Math.max(round.min_t, ev.t - 0.6);
        $('rpScrubber').value = String(curT);
        drawFrame();
        updatePlayersHud(round);
      });
      box.appendChild(div);
    });
  }

  function updatePlayersHud(round) {
    const container = $('rpPlayersHud');
    container.replaceChildren();
    const sorted = [...bundle.participants].sort((a, b) => {
      const sa = seatSide(round, a.seat) === 'T' ? 0 : 1;
      const sb = seatSide(round, b.seat) === 'T' ? 0 : 1;
      return sa - sb || a.seat - b.seat;
    });
    sorted.forEach(p => {
      const pt = sampleTrack(round.tracks[String(p.seat)], curT);
      const alive = pt ? pt[6] === 1 : false;
      const hp = pt ? pt[5] : 0;
      const wpn = pt ? (pt[8] || '') : '';
      const side = seatSide(round, p.seat);
      const row = document.createElement('div');
      row.className = 'ph-row' + (!alive ? ' dead' : '') + (p.seat === ownSeat ? ' is-mine' : '');
      const dot = document.createElement('span');
      dot.className = 'dot ' + (p.seat === ownSeat ? 'dot-own' : side === 'T' ? 'dot-t' : 'dot-ct');
      const meta = extraMetaBySeat[p.seat] || {};
      const name = document.createElement('span');
      name.className = 'ph-name';
      name.textContent = (p.seat === ownSeat ? '★ ' : '') + p.display_name;
      const rankTxt = meta.rank != null ? `#${meta.rank}` : '';
      const ratingVal = meta.rating != null ? Math.round(meta.rating) : (meta.rating_change?.before != null ? Math.round(meta.rating_change.before) : null);
      const ratingTxt = ratingVal != null ? `${ratingVal}` : '';
      name.title = `${p.display_name} (Seat ${p.seat}, ${side}${rankTxt ? ', ' + rankTxt : ''}${ratingTxt ? ', Rating ' + ratingTxt : ''})`;
      const rankEl = document.createElement('span');
      rankEl.className = 'ph-rank';
      rankEl.textContent = rankTxt ? `${rankTxt}${ratingTxt ? '·' + ratingTxt : ''}` : (ratingTxt || '—');
      if (meta.rank != null && meta.rank <= 20) rankEl.style.color = '#fde047';
      else if (meta.rank != null && meta.rank <= 50) rankEl.style.color = '#7dd3fc';
      const wpnEl = document.createElement('span');
      wpnEl.className = 'ph-wpn';
      wpnEl.textContent = wpn.replace(/^weapon_/, '');
      const hpEl = document.createElement('span');
      hpEl.className = 'ph-hp';
      hpEl.style.color = !alive ? '#64748b' : hp > 50 ? '#4ade80' : hp > 20 ? '#fbbf24' : '#f43f5e';
      hpEl.textContent = alive ? `${hp}` : '☠';
      row.append(dot, name, rankEl, wpnEl, hpEl);
      container.appendChild(row);
    });
  }

  function drawFrame() {
    if (!bundle || !bundle.rounds.length) return;
    const round = bundle.rounds[roundIdx];
    const canvas = $('replayCanvas');
    const wrap = canvas.parentElement;
    const rect = wrap.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.max(200, Math.floor(rect.width));
    const ch = Math.max(160, Math.floor(rect.height));
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = '#070b11';
    ctx.fillRect(0, 0, cw, ch);

    const proj = makeProjection(bundle.map.bounds, cw, ch);
    const showTrails = $('rpTrails').checked;
    const showCones = $('rpCones').checked;
    const showLabels = $('rpLabels').checked;

    // Subtle tactical grid
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < cw; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke(); }
    for (let y = 0; y < ch; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke(); }

    // Draw map polygons (floors -> ramps -> cover -> walls)
    const layers = { floor: [], ramp: [], cover: [], wall: [] };
    for (const poly of bundle.map.polygons) {
      (layers[poly.kind] || layers.cover).push(poly);
    }
    const drawPoly = (poly, fill, stroke) => {
      if (!poly.pts || poly.pts.length < 3) return;
      ctx.beginPath();
      poly.pts.forEach((p, i) => {
        const [sx, sy] = proj.xy(p[0], p[1]);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
    };
    layers.floor.forEach(p => drawPoly(p, 'rgba(30, 41, 59, 0.55)', 'rgba(71, 85, 105, 0.25)'));
    layers.ramp.forEach(p => drawPoly(p, 'rgba(51, 65, 85, 0.65)', 'rgba(148, 163, 184, 0.35)'));
    layers.cover.forEach(p => drawPoly(p, 'rgba(56, 72, 95, 0.85)', 'rgba(148, 163, 184, 0.4)'));
    layers.wall.forEach(p => drawPoly(p, 'rgba(15, 23, 42, 0.96)', 'rgba(100, 116, 139, 0.55)'));

    // CS2 Triangle maps (dust2/mirage): draw walkable nav corridors & callout labels
    const navLines = bundle.map.nav_lines || [];
    if (navLines.length) {
      ctx.beginPath();
      for (let i = 0; i < navLines.length; i++) {
        const seg = navLines[i];
        const [x0, y0] = proj.xy(seg[0], seg[1]);
        const [x1, y1] = proj.xy(seg[2], seg[3]);
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
      }
      ctx.strokeStyle = 'rgba(71, 105, 145, 0.45)';
      ctx.lineWidth = Math.max(2, proj.scale * 1.4);
      ctx.lineCap = 'round';
      ctx.stroke();
    }
    const mapLabels = bundle.map.labels || [];
    if (mapLabels.length) {
      ctx.font = '700 10px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const lb of mapLabels) {
        const [lx, ly] = proj.xy(lb.x, lb.y);
        ctx.fillStyle = 'rgba(148, 163, 184, 0.68)';
        ctx.fillText(lb.text, lx, ly);
      }
    }

    // Bomb sites
    for (const site of bundle.map.sites || []) {
      const [sx, sy] = proj.xy(site.x, site.y);
      const rad = Math.max(12, (site.r || 3.5) * proj.scale);
      ctx.beginPath();
      ctx.arc(sx, sy, rad, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(250, 204, 21, 0.08)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.45)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(250, 204, 21, 0.78)';
      ctx.font = '700 13px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(site.name).toUpperCase(), sx, sy);
    }

    // Full / played trails
    if (showTrails) {
      for (const p of bundle.participants) {
        const track = round.tracks[String(p.seat)];
        if (!track || track.length < 2) continue;
        const side = seatSide(round, p.seat);
        const isOwn = p.seat === ownSeat;
        ctx.beginPath();
        let penDown = false;
        for (let i = 0; i < track.length; i++) {
          const pt = track[i];
          if (pt[0] > curT) break;
          if (pt[6] === 0) { penDown = false; continue; }
          const [sx, sy] = proj.xy(pt[1], pt[2]);
          if (!penDown) { ctx.moveTo(sx, sy); penDown = true; }
          else ctx.lineTo(sx, sy);
        }
        ctx.strokeStyle = isOwn ? 'rgba(250, 204, 21, 0.75)' : side === 'T' ? 'rgba(251, 146, 60, 0.3)' : 'rgba(56, 189, 248, 0.3)';
        ctx.lineWidth = isOwn ? 2.4 : 1.25;
        ctx.stroke();
      }
    }

    // Recent firing tracers & kill markers
    for (const ev of round.events) {
      const dt = curT - ev.t;
      if (dt < 0) continue;
      if (ev.type === 'fire' && dt <= 0.35 && ev.from) {
        const [x0, y0] = proj.xy(ev.from[0], ev.from[1]);
        let x1, y1;
        if (ev.to) {
          [x1, y1] = proj.xy(ev.to[0], ev.to[1]);
        } else {
          const pt = sampleTrack(round.tracks[String(ev.player)], ev.t);
          if (pt) {
            [x1, y1] = proj.xy(ev.from[0] + Math.cos(pt[4]) * 18, ev.from[1] - Math.sin(pt[4]) * 18);
          }
        }
        if (x1 != null) {
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          const alpha = Math.max(0.1, 1 - dt / 0.35);
          ctx.strokeStyle = ev.player === ownSeat ? `rgba(250, 204, 21, ${alpha})` : `rgba(248, 250, 252, ${alpha * 0.65})`;
          ctx.lineWidth = ev.player === ownSeat ? 2.2 : 1.2;
          ctx.stroke();
        }
      }
      if (ev.type === 'kill' && dt <= 6.0) {
        const vpt = sampleTrack(round.tracks[String(ev.target)], ev.t);
        if (vpt) {
          const [kx, ky] = proj.xy(vpt[1], vpt[2]);
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.85)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(kx - 5, ky - 5); ctx.lineTo(kx + 5, ky + 5);
          ctx.moveTo(kx + 5, ky - 5); ctx.lineTo(kx - 5, ky + 5);
          ctx.stroke();
        }
      }
    }

    // Bomb indicator
    const bomb = sampleBomb(round.bomb, curT);
    const hudBomb = $('hudBomb');
    if (bomb) {
      const [bx, by] = proj.xy(bomb[1], bomb[2]);
      const planted = bomb[3] === 'planted';
      if (planted) {
        const pulse = 7 + Math.sin(performance.now() / 160) * 3;
        ctx.beginPath();
        ctx.arc(bx, by, pulse, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(bx, by, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ef4444';
      ctx.fill();
      hudBomb.hidden = false;
      hudBomb.textContent = planted ? '💣 已安包' : bomb[3] === 'carried' ? '💣 携带中' : `💣 ${bomb[3] || '炸弹'}`;
    } else {
      hudBomb.hidden = true;
    }

    // Draw 10 players
    for (const p of bundle.participants) {
      const pt = sampleTrack(round.tracks[String(p.seat)], curT);
      if (!pt || pt[6] === 0) continue;
      const [px, py] = proj.xy(pt[1], pt[2]);
      const side = seatSide(round, p.seat);
      const isOwn = p.seat === ownSeat;
      const fill = side === 'T' ? '#fb923c' : '#38bdf8';

      // Sight cone
      if (showCones) {
        const [fx, fy] = proj.xy(pt[1] + Math.cos(pt[4]) * 6.5, pt[2] - Math.sin(pt[4]) * 6.5);
        const angle = Math.atan2(fy - py, fx - px);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.arc(px, py, isOwn ? 42 : 28, angle - 0.35, angle + 0.35);
        ctx.closePath();
        ctx.fillStyle = isOwn ? 'rgba(250, 204, 21, 0.22)' : side === 'T' ? 'rgba(251, 146, 60, 0.12)' : 'rgba(56, 189, 248, 0.12)';
        ctx.fill();
      }

      // Own glowing ring
      if (isOwn) {
        ctx.beginPath();
        ctx.arc(px, py, 11, 0, Math.PI * 2);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2.2;
        ctx.stroke();
      }

      // Player body circle
      ctx.beginPath();
      ctx.arc(px, py, isOwn ? 6.5 : 5.2, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = '#0b1017';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label with subtle pill background & canvas edge clamping
      if (showLabels || isOwn) {
        const text = playerLabel(p.seat);
        ctx.font = isOwn ? '700 11px -apple-system, sans-serif' : '500 10px -apple-system, sans-serif';
        const tw = ctx.measureText(text).width;
        const lx = Math.max(tw / 2 + 6, Math.min(cw - tw / 2 - 6, px));
        const ly = Math.max(16, py - (isOwn ? 13 : 9));
        ctx.fillStyle = 'rgba(7, 11, 17, 0.72)';
        ctx.fillRect(lx - tw / 2 - 3, ly - 11, tw + 6, 13);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillStyle = isOwn ? '#fde047' : 'rgba(226, 232, 240, 0.92)';
        ctx.fillText(text, lx, ly + 1);
      }
    }

    // Update top HUD chips & timeline readouts
    const timeStr = `${curT >= 0 ? '+' : ''}${curT.toFixed(1)}s`;
    const durStr = `${round.duration.toFixed(1)}s`;
    $('hudRoundInfo').textContent = `R${round.round_num} / ${bundle.rounds.length} · ${bundle.map.id}`;
    $('hudClock').textContent = `${timeStr} / ${durStr}`;
    $('rpCurTime').textContent = timeStr;
    $('rpDurTime').textContent = durStr;
    if (Array.isArray(round.score) && round.score.length >= 2) {
      $('hudScore').textContent = `Team0 ${round.score[0]} : ${round.score[1]} Team1`;
    }
  }

  function updateTransportUI() {
    const playBtn = $('rpPlayBtn');
    const revBtn = $('rpReverseBtn');
    const dirChip = $('hudDirection');
    playBtn.textContent = playDir === 1 ? '⏸ 暂停' : '▶ 播放';
    revBtn.textContent = playDir === -1 ? '⏸ 停倒放' : '◀ 倒放';
    revBtn.classList.toggle('active', playDir === -1);
    if (dirChip) {
      dirChip.hidden = playDir !== -1;
    }
    if (bundle) {
      $('rpPrevRound').disabled = roundIdx <= 0;
      $('rpNextRound').disabled = roundIdx >= bundle.rounds.length - 1;
    }
  }

  function selectRound(idx, startAtEnd = false) {
    roundIdx = Math.max(0, Math.min(bundle.rounds.length - 1, idx));
    const round = bundle.rounds[roundIdx];
    curT = startAtEnd ? round.duration : Math.max(round.min_t, 0);
    const scrubber = $('rpScrubber');
    scrubber.min = String(round.min_t);
    scrubber.max = String(round.duration);
    scrubber.value = String(curT);
    renderRoundBar();
    renderScrubberMarkers(round);
    renderEventsList(round);
    updatePlayersHud(round);
    updateTransportUI();
    drawFrame();
  }

  function stepTime(deltaSec) {
    if (!bundle || !bundle.rounds.length) return;
    const round = bundle.rounds[roundIdx];
    const nextT = curT + deltaSec;
    if (nextT < round.min_t && roundIdx > 0) {
      selectRound(roundIdx - 1, true);
      return;
    }
    if (nextT > round.duration && roundIdx + 1 < bundle.rounds.length) {
      selectRound(roundIdx + 1, false);
      return;
    }
    curT = Math.max(round.min_t, Math.min(round.duration, nextT));
    $('rpScrubber').value = String(curT);
    drawFrame();
    updatePlayersHud(round);
  }

  function tick(ts) {
    if (playDir === 0 || !bundle) return;
    if (!lastRaf) lastRaf = ts;
    const dt = Math.min(0.1, (ts - lastRaf) / 1000) * speed * playDir;
    lastRaf = ts;
    const round = bundle.rounds[roundIdx];
    curT += dt;
    if (playDir > 0 && curT >= round.duration) {
      if (roundIdx + 1 < bundle.rounds.length) {
        selectRound(roundIdx + 1, false);
      } else {
        curT = round.duration;
        setPlayDir(0);
      }
    } else if (playDir < 0 && curT <= round.min_t) {
      if (roundIdx > 0) {
        selectRound(roundIdx - 1, true);
      } else {
        curT = round.min_t;
        setPlayDir(0);
      }
    }
    $('rpScrubber').value = String(curT);
    drawFrame();
    if (ts - hudTick > 180) {
      hudTick = ts;
      updatePlayersHud(bundle.rounds[roundIdx]);
    }
    if (playDir !== 0) {
      rafId = requestAnimationFrame(tick);
    }
  }

  function setPlayDir(dir) {
    playDir = dir;
    lastRaf = 0;
    updateTransportUI();
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (playDir !== 0) {
      rafId = requestAnimationFrame(tick);
    }
  }

  function stopPolling() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  async function pollStream(jobId) {
    if (activeJobId !== jobId || $('replayModal').hidden) return;
    try {
      const haveParam = [...readyRoundsSet].join(',');
      const resp = await fetch(`/api/replay-poll/${encodeURIComponent(jobId)}?have=${encodeURIComponent(haveParam)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const res = await resp.json();
      if (activeJobId !== jobId) return;

      const progEl = $('hudStreamProgress');
      if (res.new_rounds) {
        for (const [riStr, payload] of Object.entries(res.new_rounds)) {
          const ri = Number(riStr);
          const targetRound = bundle.rounds.find(r => r.index === ri);
          if (targetRound) {
            targetRound.tracks = payload.tracks;
            targetRound.bomb = payload.bomb;
          }
        }
      }
      if (Array.isArray(res.rounds_ready)) {
        res.rounds_ready.forEach(ri => readyRoundsSet.add(ri));
      }

      const done = res.chunks_done || 0;
      const total = Math.max(1, res.chunks_total || 1);
      const pct = Math.min(100, Math.round((done / total) * 100));

      if (res.status === 'ready' || res.status === 'cached') {
        bundle.rounds.forEach(r => readyRoundsSet.add(r.index));
        progEl.hidden = false;
        progEl.textContent = '✓ 全部已缓存';
        setTimeout(() => { if (activeJobId === jobId) progEl.hidden = true; }, 2500);
        renderRoundBar();
        drawFrame();
        if (!autoStarted) {
          autoStarted = true;
          selectRound(roundIdx, false);
          setPlayDir(1);
        }
        return;
      }

      if (res.status === 'error') {
        progEl.hidden = false;
        progEl.textContent = `⚠️ 缓存中断: ${res.error || '网络错误'}`;
        return;
      }

      progEl.hidden = false;
      progEl.textContent = `⚡ 缓存中 ${pct}% (${done}/${total}) · 已就绪 ${readyRoundsSet.size}/${bundle.rounds.length} 回合`;
      renderRoundBar();

      // Auto-start Round 1 as soon as its frames arrive (~2-3s!)
      const curRound = bundle.rounds[roundIdx];
      const curHasFrames = curRound && curRound.tracks && Object.values(curRound.tracks).some(tr => tr.length >= 4);
      if (!autoStarted && curHasFrames) {
        autoStarted = true;
        selectRound(roundIdx, false);
        setPlayDir(1);
      } else {
        drawFrame();
      }

      pollTimer = setTimeout(() => pollStream(jobId), 1000);
    } catch (_) {
      pollTimer = setTimeout(() => pollStream(jobId), 2000);
    }
  }

  async function openReplay(jobId, opts = {}) {
    stopPolling();
    activeJobId = jobId;
    extraMetaBySeat = {};
    for (const pm of (opts.participants || [])) {
      if (pm && pm.seat != null) extraMetaBySeat[pm.seat] = pm;
    }
    readyRoundsSet = new Set();
    autoStarted = false;
    const modal = $('replayModal');
    modal.hidden = false;
    $('replayLoading').hidden = false;
    $('replayWorkspace').hidden = true;
    $('hudStreamProgress').hidden = true;
    $('replayLoadingText').textContent = '正在加载地图与对局事件（约 1 秒）…';
    $('replayTitle').textContent = `2D 战术回放 · ${opts.mapId || jobId.slice(0, 8)}`;
    $('replaySub').textContent = `${opts.accountName || ''} · Job ${jobId}`;
    $('replayOpenAGX').href = `https://agx.cool/#/matches/${jobId}`;
    setPlayDir(0);

    try {
      const initResp = await fetch(`/api/replay-init/${encodeURIComponent(jobId)}`);
      if (!initResp.ok) {
        const err = await initResp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${initResp.status}`);
      }
      const initData = await initResp.json();
      if (activeJobId !== jobId) return;

      if (initData.cached) {
        const resp = await fetch(`/api/replay/${encodeURIComponent(jobId)}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        bundle = await resp.json();
        bundle.rounds.forEach(r => readyRoundsSet.add(r.index));
        ownSeat = opts.ownSeat != null ? opts.ownSeat : (bundle.participants.find(p => p.is_mine)?.seat ?? null);
        $('replayTitle').textContent = `2D 战术回放 · ${bundle.map.id}`;
        $('replayLoading').hidden = true;
        $('replayWorkspace').hidden = false;
        autoStarted = true;
        selectRound(0, false);
        setPlayDir(1);
        return;
      }

      // Progressive mode: show map & events immediately, stream rounds in background!
      bundle = initData.skeleton;
      bundle.rounds.forEach(r => {
        if (!r.tracks) r.tracks = Object.fromEntries(Array.from({ length: 10 }, (_, i) => [String(i), []]));
        if (!r.bomb) r.bomb = [];
      });
      (initData.rounds_ready || []).forEach(ri => readyRoundsSet.add(ri));
      ownSeat = opts.ownSeat != null ? opts.ownSeat : (bundle.participants.find(p => p.is_mine)?.seat ?? null);
      $('replayTitle').textContent = `2D 战术回放 · ${bundle.map.id}`;
      $('replayLoading').hidden = true;
      $('replayWorkspace').hidden = false;
      $('hudStreamProgress').hidden = false;
      $('hudStreamProgress').textContent = `⚡ 正在高速拉取第 1 回合帧流 (0/${initData.chunks_total || '?'})…`;
      selectRound(0, false);
      pollStream(jobId);
    } catch (err) {
      $('replayLoadingText').textContent = `回放加载失败：${err.message}（可点击右上角在 AGX 官方平台直接观看 3D 回放）`;
    }
  }

  function closeReplay() {
    stopPolling();
    activeJobId = null;
    setPlayDir(0);
    $('replayModal').hidden = true;
  }

  window.addEventListener('DOMContentLoaded', () => {
    $('replayClose').addEventListener('click', closeReplay);
    $('replayModal').addEventListener('click', e => { if (e.target === $('replayModal')) closeReplay(); });
    $('rpPlayBtn').addEventListener('click', () => setPlayDir(playDir === 1 ? 0 : 1));
    $('rpReverseBtn').addEventListener('click', () => setPlayDir(playDir === -1 ? 0 : -1));
    $('rpStepBack').addEventListener('click', () => stepTime(-5));
    $('rpStepFwd').addEventListener('click', () => stepTime(5));
    $('rpPrevRound').addEventListener('click', () => { if (bundle && roundIdx > 0) selectRound(roundIdx - 1, false); });
    $('rpNextRound').addEventListener('click', () => { if (bundle && roundIdx + 1 < bundle.rounds.length) selectRound(roundIdx + 1, false); });
    $('rpSpeed').addEventListener('change', e => { speed = Number(e.target.value) || 1; });
    $('rpScrubber').addEventListener('input', e => {
      curT = Number(e.target.value);
      drawFrame();
      if (bundle) updatePlayersHud(bundle.rounds[roundIdx]);
    });
    ['rpTrails', 'rpCones', 'rpLabels'].forEach(id => $(id).addEventListener('change', drawFrame));
    window.addEventListener('resize', () => { if (!$('replayModal').hidden) drawFrame(); });
    window.addEventListener('keydown', e => {
      if ($('replayModal').hidden) return;
      if (e.key === 'Escape') { closeReplay(); return; }
      if (e.key === ' ') { e.preventDefault(); setPlayDir(playDir === 1 ? 0 : 1); return; }
      if (e.key === 'r' || e.key === 'R') { e.preventDefault(); setPlayDir(playDir === -1 ? 0 : -1); return; }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepTime(e.shiftKey ? -0.5 : -2); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); stepTime(e.shiftKey ? 0.5 : 2); return; }
      if (e.key === '[' && bundle && roundIdx > 0) { selectRound(roundIdx - 1, false); return; }
      if (e.key === ']' && bundle && roundIdx + 1 < bundle.rounds.length) { selectRound(roundIdx + 1, false); }
    });
  });

  window.FPSReplayTheater = { open: openReplay, close: closeReplay };
})();
