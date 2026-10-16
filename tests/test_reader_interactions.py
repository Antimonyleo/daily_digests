"""Execute small browser behaviors without a browser or third-party JS packages."""

import shutil
import subprocess
from pathlib import Path

import pytest


TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "digest_web.html.j2"


def _run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for browser-script regression checks")
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_reading_progress_updates_when_digest_layout_changes():
    source = TEMPLATE.read_text()
    script = source.split('  const progressBar =')[1].split('  function setActive')[0]
    _run_js('''
        const assert = require('node:assert/strict');
        let height = 1000;
        const bar = { classList: { contains: () => false, toggle: () => {} },
            setAttribute: (key, value) => { bar[key] = value; } };
        const card = { getBoundingClientRect: () => ({top: 100, height}) };
        const document = {
            getElementById: (id) => id === 'reading-progress' ? bar : {style: {}},
            querySelector: () => card,
        };
        let frame, observer;
        const window = {scrollY: 400, innerHeight: 800, addEventListener: () => {},
            requestAnimationFrame: (callback) => { frame = callback; return 1; }};
        class ResizeObserver {
            constructor(callback) { observer = callback; }
            observe() {}
        }
    ''' + 'const progressBar =' + script + '''
        frame();
        assert.equal(bar['aria-valuenow'], '17');
        height = 500;
        assert.equal(typeof observer, 'function', 'layout changes must refresh progress');
        observer();
        frame();
        assert.equal(bar['aria-valuenow'], '100');
    ''')


def test_failed_brew_retry_preserves_reader_selected_serving():
    source = TEMPLATE.with_name("run.html.j2").read_text()
    script = source[source.index("  async function startBrew()"):source.index("  if (RESUME)")]
    _run_js('''
        const assert = require('node:assert/strict');
        const RESTART_URL = '/run?reading_mode=usual&autostart=1';
        const RESUME_URL = '/run?reading_mode=usual&resume=1&run_id=abcdef123456';
        const RUN_ID = 'abcdef123456', CSRF_TOKEN = 'fixture';
        let restartUrl = RESTART_URL;
        const control = () => ({events:{},addEventListener(k,cb){this.events[k]=cb;}});
        const form=control(), retryButton=control(), reconnectButton=control(), cancelButton=control();
        const newBrewLink={href:RESTART_URL};
        const brewButton={disabled:false}, choicePanel={}, progressPanel={};
        const window={location:{href:'http://127.0.0.1:8799/run'},history:{replaceState(){}}};
        class FormData {get(){return 'minimal';}}
        const fetch=async()=>({ok:false,status:503});
        const showError=()=>{};
        const connectStream=()=>assert.fail('failed start must not create a status stream');
    ''' + script + '''
        (async()=>{
            await startBrew();
            retryButton.events.click();
            assert.equal(new URL(window.location.href,'http://fixture.invalid').searchParams.get('reading_mode'),'minimal');
            assert.equal(new URL(newBrewLink.href,'http://fixture.invalid').searchParams.get('reading_mode'),'minimal');
        })().catch(error=>{console.error(error);process.exitCode=1;});
    ''')


@pytest.mark.parametrize('leaves, expected', [(4, 'Lab joke: second'), (10, 'complete')])
def test_pip_restores_note_with_saved_leaf_count(leaves, expected):
    source = TEMPLATE.read_text()
    script = source.split('  const teaText =')[1].split('  function clampPipPosition')[0]
    script = script.replace('{{ digest_id|tojson }}', '"2026-09-11"')
    _run_js('''
        const assert = require('node:assert/strict');
        const TEA_NOTES = ['Science fact: first', 'Lab joke: second', 'Science fact: third'];
        const nodes = {};
        const document = {getElementById: (id) => nodes[id] ||= {
            textContent: TEA_NOTES[0], dataset: {}, setAttribute: () => {},
        }};
        const window = {matchMedia: () => ({matches: false})};
    ''' + f'const localStorage = {{getItem: () => "{leaves}"}};'
        + 'const teaText =' + script + '''
        syncTeaBreakState();
    ''' + ('''
        assert.equal(teaText.textContent, teaBreakOver);
        assert.equal(teaButton.disabled, true);
    ''' if expected == 'complete' else '''
        assert.equal(teaText.textContent, TEA_NOTES[1]);
        assert.equal(teaBreak.dataset.noteKind, 'joke');
    '''))


def _run_pip_js(check, *, leaves=0, reduced=False, single_note=False):
    source = TEMPLATE.read_text()
    script = 'const teaText =' + source.split('  const teaText =')[1].split('  const progressBar =')[0]
    script = script.replace('{{ digest_id|tojson }}', '"preview"')
    _run_js(f'const initialLeaves = {leaves}, initiallyReduced = {str(reduced).lower()};' + '''
        const assert = require('node:assert/strict');
        const TEA_NOTES = ['Science fact: first', 'Lab joke: second', 'Science fact: third'];
    ''' + ('TEA_NOTES.splice(1);' if single_note else '') + '''
        let now = 0, nextTimer = 0, visible = true, intersect;
        const timers = new Map(), nodes = {}, documentEvents = {}, windowEvents = {};
        Math.random = () => 0;
        function tick(ms) {
            const end = now + ms;
            while (true) {
                const next = [...timers].sort((a,b) => a[1].at - b[1].at)[0];
                if (!next || next[1].at > end) break;
                now = next[1].at;
                timers.delete(next[0]);
                next[1].callback();
            }
            now = end;
        }
        function node(id) {
            if (nodes[id]) return nodes[id];
            const classes = new Set();
            return nodes[id] = {
                textContent: TEA_NOTES[0], dataset: {}, events: {}, disabled: false,
                style: {setProperty(k,v) {this[k]=v;}, removeProperty(k) {delete this[k];}},
                classList: {add: (...names) => names.forEach(n => classes.add(n)),
                    remove: (...names) => names.forEach(n => classes.delete(n)),
                    contains: n => classes.has(n)},
                setAttribute(k,v) {this[k]=v;}, removeAttribute(k) {delete this[k];},
                addEventListener(k,cb) {this.events[k]=cb;},
                setPointerCapture() {this.captured=true;}, hasPointerCapture() {return this.captured;},
                releasePointerCapture() {this.captured=false;},
                offsetWidth:100, offsetHeight:100,
                getClientRects: () => visible ? [{}] : [],
                getBoundingClientRect: () => ({left:20, top:20, right:120, bottom:120, width:100, height:100}),
                querySelector: selector => node(selector), closest: selector => node(selector),
            };
        }
        const motion = {matches: initiallyReduced, addEventListener(k,cb) {this.change=cb;}};
        const document = {hidden: false, getElementById: node,
            addEventListener: (k,cb) => documentEvents[k]=cb};
        const window = {innerWidth:1000, innerHeight:800,
            matchMedia: q => q.includes('reduced-motion') ? motion : {matches:false, addEventListener() {}},
            addEventListener: (k,cb) => windowEvents[k]=cb, clearTimeout: id => timers.delete(id),
            setTimeout: (callback,delay) => {timers.set(++nextTimer, {at:now+delay, callback}); return nextTimer;}};
        class IntersectionObserver {
            constructor(cb) {intersect=cb;}
            observe() {}
        }
        const saved = {'dailydigest-tea-leaves:preview': String(initialLeaves)};
        const localStorage = {getItem: k => saved[k], setItem: (k,v) => saved[k]=v, removeItem: k => delete saved[k]};
    ''' + script + check)


def test_pip_feeding_reacts_after_impact_and_ignores_repeated_clicks():
    _run_pip_js('''
        teaButton.events.click();
        assert.equal(saved[teaLeafStorageKey], '1');
        assert.equal(teaText.textContent, TEA_NOTES[0], 'keep the old note until impact');
        assert.equal(teaBreak.classList.contains('is-sipping'), false);
        teaButton.events.click();
        assert.equal(saved[teaLeafStorageKey], '1');
        tick(LEAF_FALL_MS - 1);
        assert.equal(teaBreak.classList.contains('is-sipping'), false);
        tick(1);
        assert.equal(teaBreak.classList.contains('is-sipping'), true);
        assert.equal(teaText.textContent, TEA_NOTES[1]);
        tick(1000);
        assert.equal(leafBusy, false);
        assert.equal(teaBreak.classList.contains('is-sipping'), false);
        teaButton.events.click();
        assert.equal(saved[teaLeafStorageKey], '2');
    ''')


@pytest.mark.parametrize('phase_ms', [100, 600, 950])
@pytest.mark.parametrize('interruption', ['hidden', 'pip-hidden', 'offscreen', 'reduced'])
def test_pip_interrupted_feeding_settles_note_and_releases_controls(phase_ms, interruption):
    interrupt = {
        'hidden': "document.hidden=true; documentEvents.visibilitychange();",
        'pip-hidden': "visible=false; documentEvents.pipvisibilitychange();",
        'offscreen': "visible=false; intersect([{isIntersecting:false}]);",
        'reduced': "motion.matches=true; motion.change();",
    }[interruption]
    _run_pip_js('teaButton.events.click();' + f'tick({phase_ms});' + interrupt + '''
        assert.equal(teaText.textContent, TEA_NOTES[1]);
        assert.equal(saved[teaLeafStorageKey], '1');
        assert.equal(leafBusy, false);
        assert.equal(timers.size, 0);
        for (const name of ['is-feeding', 'is-sipping', 'is-regrowing', 'is-blinking', 'is-steaming']) {
            assert.equal(teaBreak.classList.contains(name), false, name);
        }
    ''')


def test_pip_final_leaf_lands_before_disabling_and_reduced_motion_is_immediate():
    _run_pip_js('''
        teaButton.events.click();
        assert.equal(teaButton.disabled, false, 'last leaf must remain visible until impact');
        tick(LEAF_FALL_MS);
        assert.equal(teaButton.disabled, true);
        assert.equal(teaText.textContent, teaBreakOver);
        tick(1000);
        teaButton.events.click();
        assert.equal(saved[teaLeafStorageKey], '10');
    ''', leaves=9)
    _run_pip_js('''
        teaButton.events.click();
        assert.equal(teaButton.disabled, true);
        assert.equal(teaText.textContent, teaBreakOver);
        assert.equal(saved[teaLeafStorageKey], '10');
        assert.equal(leafBusy, false);
        assert.equal(timers.size, 0);
    ''', leaves=9, reduced=True)


def test_pip_single_note_can_be_fed_and_idle_resumes_after_a_rest():
    _run_pip_js('''
        teaButton.events.click();
        tick(2000);
        assert.equal(saved[teaLeafStorageKey], '1');
        assert.equal(teaText.textContent, TEA_NOTES[0]);
        document.hidden = true;
        documentEvents.visibilitychange();
        assert.equal(timers.size, 0);
        document.hidden = false;
        documentEvents.visibilitychange();
        tick(7999);
        assert.equal(teaBreak.classList.contains('is-steaming'), false);
        tick(1);
        assert.equal(teaBreak.classList.contains('is-steaming'), true);
        assert.equal(teaText.textContent, TEA_NOTES[0], 'idle gestures must not announce new text');
    ''', single_note=True)


def test_pip_idle_alternates_steam_with_curiosity_without_consuming_leaves():
    _run_pip_js('''
        tick(8000);
        assert.equal(teaBreak.classList.contains('is-steaming'), true);
        assert.equal(teaBreak.classList.contains('is-curious'), false);
        tick(2000);
        assert.equal(teaBreak.classList.contains('is-steaming'), false);
        tick(8000);
        assert.equal(teaBreak.classList.contains('is-steaming'), true);
        tick(10000);
        assert.equal(teaBreak.classList.contains('is-curious'), true);
        assert.equal(teaBreak.classList.contains('is-steaming'), false);
        assert.equal(teaText.textContent, TEA_NOTES[0]);
        assert.equal(saved[teaLeafStorageKey], '0');
        tick(2000);
        assert.equal(teaBreak.classList.contains('is-curious'), false);
        assert.equal(timers.size, 1, 'one idle timer, with no overlapping gestures');
    ''')
    _run_pip_js('''
        tick(28000);
        assert.equal(teaBreak.classList.contains('is-curious'), false, 'no leaf remains to watch');
        assert.equal(teaBreak.classList.contains('is-blinking'), true);
        assert.equal(teaText.textContent, teaBreakOver);
        assert.equal(saved[teaLeafStorageKey], '10');
    ''', leaves=10)


@pytest.mark.parametrize('idle_ms', [8100, 28100])
@pytest.mark.parametrize('interruption', ['feed', 'drag', 'pointer', 'hidden', 'offscreen', 'reduced'])
def test_pip_idle_gives_way_to_interactions_and_visibility(idle_ms, interruption):
    interrupt = {
        'feed': 'teaButton.events.click(); assert.equal(leafBusy, true);',
        'drag': '''pipDock.matches=true;
            teaPet.events.pointerdown({button:0, pointerId:1, clientX:30, clientY:30, preventDefault() {}});''',
        'pointer': '''finePointer.matches=true;
            teaPet.closest('.pip-stage').events.pointermove({clientX:50, clientY:50});''',
        'hidden': 'document.hidden=true; documentEvents.visibilitychange();',
        'offscreen': 'visible=false; intersect([{isIntersecting:false}]);',
        'reduced': 'motion.matches=true; motion.change();',
    }[interruption]
    _run_pip_js(f'tick({idle_ms});' + interrupt + '''
        for (const name of ['is-steaming', 'is-curious', 'is-blinking']) {
            assert.equal(teaBreak.classList.contains(name), false, name);
        }
    ''' + ('''assert.equal(timers.size, 0);''' if interruption in {'hidden', 'offscreen', 'reduced'} else ''))


@pytest.mark.parametrize('interruption', ['hidden', 'blur', 'pointercancel', 'lostpointercapture'])
def test_pip_interrupted_drag_releases_capture_and_allows_feeding(interruption):
    interrupt = {
        'hidden': 'document.hidden=true; documentEvents.visibilitychange();',
        'blur': 'windowEvents.blur();',
        'pointercancel': 'teaPet.events.pointercancel();',
        'lostpointercapture': 'teaPet.captured=false; teaPet.events.lostpointercapture();',
    }[interruption]
    _run_pip_js('''
        pipDock.matches = true;
        teaPet.events.pointerdown({button:0, pointerId:1, clientX:30, clientY:30, preventDefault() {}});
        assert.equal(teaBreak.classList.contains('is-dragging'), true);
    ''' + interrupt + '''
        assert.equal(pipDrag, null);
        assert.equal(teaPet.captured, false);
        assert.equal(teaBreak.classList.contains('is-dragging'), false);
        document.hidden = false;
        documentEvents.visibilitychange();
        teaButton.events.click();
        assert.equal(saved[teaLeafStorageKey], '1');
    ''')


@pytest.mark.parametrize('storage_available', [True, False])
def test_display_preferences_and_menu_keyboard_dismissal(storage_available):
    import re

    base = TEMPLATE.with_name('_base.html.j2').read_text()
    base = base.replace('{% include "_rating_queue.js.j2" %}', TEMPLATE.with_name('_rating_queue.js.j2').read_text())
    scripts = re.findall(r'<script>(.*?)</script>', base, re.DOTALL)
    _run_js(f'const storageAvailable = {str(storage_available).lower()};' + '''
        const assert = require('node:assert/strict');
        const nodes = {};
        const docEvents = {};
        const saved = {'dailydigest-theme': 'dark', 'dailydigest-display': 'compact',
            'dailydigest-pip': 'hidden'};
        const localStorage = {
            getItem: key => {if (!storageAvailable) throw Error('denied'); return saved[key];},
            setItem: (key,value) => {if (!storageAvailable) throw Error('denied'); saved[key]=value;},
        };
        const document = {
            documentElement: {dataset: {}},
            getElementById: id => nodes[id] ||= {hidden: true, open: true, events: {},
                setAttribute(key,value) {this[key]=value;},
                addEventListener(type, callback) {this.events[type]=callback;},
                querySelector() {return document.getElementById('display-menu-toggle');},
                contains(target) {return Boolean(target.insideMenu);},
                focus() {this.focused=true;},
            },
            querySelector: () => ({}),
            addEventListener: (type,callback) => docEvents[type]=callback,
            dispatchEvent: () => {},
        };
        const window = {matchMedia: () => ({matches:false}), addEventListener: () => {}};
        const navigator = {};
    ''' + '\n'.join(scripts) + '''
        assert.equal(document.documentElement.dataset.theme, storageAvailable ? 'dark' : 'light');
        assert.equal(nodes['display-toggle']['aria-pressed'], storageAvailable ? 'true' : 'false');
        assert.equal(nodes['display-toggle'].textContent, 'Compact view');
        assert.equal(nodes['pip-toggle']['aria-pressed'], storageAvailable ? 'false' : 'true');
        nodes['pip-toggle'].events.click();
        if (storageAvailable) assert.equal(saved['dailydigest-pip'], 'visible');
        assert.equal(document.documentElement.dataset.pip, storageAvailable ? 'visible' : 'hidden');
        assert.equal(nodes['pip-toggle']['aria-pressed'], storageAvailable ? 'true' : 'false');
        nodes['display-toggle'].events.click();
        if (storageAvailable) assert.equal(saved['dailydigest-display'], 'comfortable');
        assert.equal(document.documentElement.dataset.display, storageAvailable ? 'comfortable' : 'compact');
        const menu = nodes['display-menu'];
        docEvents.pointerdown({target:{insideMenu:true}});
        docEvents.focusin({target:{insideMenu:true}});
        assert.equal(menu.open, true);
        menu.events.keydown({key:'Escape', preventDefault() {}});
        assert.equal(menu.open, false);
        assert.equal(nodes['display-menu-toggle'].focused, true);
        menu.open = true;
        docEvents.pointerdown({target:{}});
        assert.equal(menu.open, false);
        menu.open = true;
        docEvents.focusin({target:{}});
        assert.equal(menu.open, false);
    ''')
