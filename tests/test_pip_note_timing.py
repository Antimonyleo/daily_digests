"""Pip chooses one note per accepted leaf; animation never rotates its text.

Runs the shipped controller against a fake DOM and a fake clock, sampling the
visible state at 0, 100, 480, 900, 1200 and 2500 ms after each gesture.
"""

from pathlib import Path

import pytest

from tests.test_reader_interactions import _run_js

PIP_JS = Path(__file__).resolve().parents[1] / "templates" / "_pip.js.j2"
SAMPLES = [0, 100, 480, 900, 1200, 2500]


def _run(check, *, prelude="", leaves=0, reduced=False):
    _run_js(f"const initialLeaves = {leaves}, initiallyReduced = {str(reduced).lower()};" + '''
        const assert = require('node:assert/strict');
        const NOTES = ['Science fact: first', 'Lab joke: second', 'Science fact: third'];
        let now = 0, nextTimer = 0, visible = true, intersect;
        const timers = new Map(), nodes = {}, documentEvents = {}, windowEvents = {};
        Math.random = () => 0;
        function tick(ms) {
            const end = now + ms;
            while (true) {
                const next = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
                if (!next || next[1].at > end) break;
                now = next[1].at; timers.delete(next[0]); next[1].callback();
            }
            now = end;
        }
        function node(id) {
            if (nodes[id]) return nodes[id];
            const classes = new Set();
            return nodes[id] = {
                id, textContent: NOTES[0], dataset: {}, events: {}, disabled: false,
                style: {setProperty(k, v) {this[k] = v;}, removeProperty(k) {delete this[k];}},
                classList: {add: (...n) => n.forEach(x => classes.add(x)),
                    remove: (...n) => n.forEach(x => classes.delete(x)), contains: n => classes.has(n)},
                setAttribute(k, v) {this[k] = v;}, removeAttribute(k) {delete this[k];},
                addEventListener(k, cb) {this.events[k] = cb;},
                setPointerCapture() {this.captured = true;}, hasPointerCapture() {return this.captured;},
                releasePointerCapture() {this.captured = false;},
                offsetWidth: 100, offsetHeight: 100,
                getClientRects: () => visible ? [{}] : [],
                getBoundingClientRect: () => ({left: 20, top: 20, right: 120, bottom: 120, width: 100, height: 100}),
                querySelector: s => node(s), closest: s => node(s),
            };
        }
        const motion = {matches: initiallyReduced, addEventListener(k, cb) {this.change = cb;}};
        const document = {hidden: false, getElementById: node,
            querySelector: s => node(s.replace(/^#/, '')), addEventListener: (k, cb) => documentEvents[k] = cb};
        const window = {innerWidth: 1000, innerHeight: 800, AbortController,
            matchMedia: q => q.includes('reduced-motion') ? motion : {matches: false, addEventListener() {}},
            addEventListener: (k, cb) => windowEvents[k] = cb, clearTimeout: id => timers.delete(id),
            setTimeout: (cb, delay) => {timers.set(++nextTimer, {at: now + delay, callback: cb}); return nextTimer;}};
        window.IntersectionObserver = class { constructor(cb) {intersect = cb;} observe() {} disconnect() {} };
        const saved = {'dailydigest-tea-leaves:day': String(initialLeaves)};
        window.localStorage = {getItem: k => saved[k] ?? null, setItem: (k, v) => saved[k] = String(v),
            removeItem: k => delete saved[k]};
        document.defaultView = window;
        const text = node('tea-note-text'), host = node('tea-break'), pet = node('tea-pet'), leaf = node('tea-another');
        const count = node('tea-leaf-count');
        let writes = [];
        let current = NOTES[0];
        Object.defineProperty(text, 'textContent', {get: () => current, set: v => {writes.push([now, v]); current = v;}});
        const state = () => ({text: current, chat: host.dataset.chat, count: count.textContent});
        function timeline() {
            const result = {};
            let at = 0;
            for (const sample of ''' + str(SAMPLES) + ''') { tick(sample - at); at = sample; result[sample] = state(); }
            return result;
        }
    ''' + prelude + PIP_JS.read_text() + '''
        const cleanup = mountPip(document, {digestId: 'day', notes: NOTES});
        writes = [];
    ''' + check)


def test_single_leaf_shows_one_new_note_atomically_and_never_rotates_it():
    _run('''
        assert.equal(host.dataset.chat, 'open', 'a first display greets with a note');
        leaf.events.click({detail: 1});
        const seen = timeline();
        for (const ms of [0, 100, 480, 900, 1200, 2500]) {
            assert.equal(seen[ms].text, NOTES[1], `note at ${ms} ms`);
            assert.equal(seen[ms].chat, 'open', `bubble at ${ms} ms`);
        }
        assert.equal(seen[0].count, '0', 'the cup counts the leaf when it lands');
        assert.equal(seen[480].count, '1');
        assert.equal(seen[2500].count, '1');
        assert.deepEqual(writes, [[0, NOTES[1]]], 'exactly one text write, at acceptance');
        assert.equal(saved['dailydigest-tea-leaves:day'], '1');
    ''')


def test_collapsed_bubble_opens_with_the_new_note_in_the_same_step():
    _run('''
        assert.equal(host.dataset.chat, 'closed', 'an explicit collapse is respected on mount');
        let openedWith = null;
        let chat = host.dataset.chat;
        Object.defineProperty(host.dataset, 'chat', {get: () => chat, set: v => {
            if (v === 'open' && openedWith === null) openedWith = current; chat = v;}});
        leaf.events.click({detail: 1});
        assert.equal(openedWith, NOTES[1], 'never opens on the previous note');
        const seen = timeline();
        assert.equal(seen[0].text, NOTES[1]);
        assert.equal(seen[2500].text, NOTES[1]);
        assert.equal(saved['dailydigest-pip-chat'], 'open');
    ''', prelude="saved['dailydigest-pip-chat'] = 'closed';")


@pytest.mark.parametrize("reduced", [False, True])
def test_fast_double_click_counts_one_leaf(reduced):
    _run('''
        leaf.events.click({detail: 1});
        tick(100);
        leaf.events.click({detail: 2});
        const seen = timeline();
        assert.equal(saved['dailydigest-tea-leaves:day'], '1');
        assert.equal(seen[2500].count, '1');
        for (const ms of [0, 100, 480, 900, 1200, 2500]) assert.equal(seen[ms].text, NOTES[1]);
        assert.deepEqual(writes.map(w => w[1]), [NOTES[1]]);
    ''', reduced=reduced)


def test_two_separate_clicks_during_a_feed_count_once():
    _run('''
        leaf.events.click({detail: 1});
        tick(300);
        leaf.events.click({detail: 1});
        timeline();
        assert.equal(saved['dailydigest-tea-leaves:day'], '1');
        leaf.events.click({detail: 1});
        const seen = timeline();
        assert.equal(saved['dailydigest-tea-leaves:day'], '2');
        assert.equal(seen[0].text, NOTES[2]);
        assert.equal(seen[2500].text, NOTES[2]);
        assert.equal(writes.length, 2);
    ''')


def test_drag_during_feed_keeps_the_note_and_count():
    _run('''
        leaf.events.click({detail: 1});
        tick(100);
        pet.events.pointerdown({button: 0, pointerId: 1, clientX: 30, clientY: 30, preventDefault() {}});
        pet.events.pointermove({pointerId: 1, clientX: 200, clientY: 200});
        assert.equal(host.classList.contains('is-dragging'), true);
        assert.equal(host.dataset.chat, 'open', 'the bubble stays readable while dragging');
        assert.equal(current, NOTES[1]);
        pet.events.pointerup({pointerId: 1});
        const seen = timeline();
        assert.equal(seen[2500].text, NOTES[1]);
        assert.equal(seen[2500].count, '1');
        assert.equal(seen[2500].chat, 'open', 'a drag is not a tap and does not collapse');
        assert.equal(saved['dailydigest-tea-leaves:day'], '1');
        assert.equal(writes.length, 1);
    ''')


def test_hide_and_reshow_during_feed_keeps_note_and_chat_state():
    _run('''
        leaf.events.click({detail: 1});
        tick(100);
        visible = false; documentEvents.pipvisibilitychange();
        visible = true; documentEvents.pipvisibilitychange();
        const seen = timeline();
        for (const ms of [0, 1200, 2500]) assert.equal(seen[ms].text, NOTES[1]);
        assert.equal(seen[2500].count, '1');
        assert.equal(seen[2500].chat, 'open');
        assert.equal(writes.length, 1);
        assert.equal(timers.size <= 1, true, 'only the idle timer may remain');
    ''')


def test_cross_page_remount_keeps_note_count_and_collapse_without_rewriting():
    _run('''
        leaf.events.click({detail: 1});
        tick(100);
        cleanup();
        // The next page (or Python takeover) mounts the same component.
        mountPip(document, {digestId: 'day', notes: NOTES});
        const seen = timeline();
        assert.equal(seen[0].text, NOTES[1]);
        assert.equal(seen[0].count, '1');
        assert.equal(seen[0].chat, 'open');
        assert.equal(writes.length, 1, 'the remount does not re-announce the same note');
        // Collapse, then a further page keeps it collapsed.
        pet.events.keydown({key: 'Escape', preventDefault() {}});
        assert.equal(saved['dailydigest-pip-chat'], 'closed');
        mountPip(document, {digestId: 'day', notes: NOTES});
        assert.equal(host.dataset.chat, 'closed');
        assert.equal(current, NOTES[1]);
    ''')


def test_reduced_motion_shows_note_and_count_immediately_with_no_timers_rotating_text():
    _run('''
        leaf.events.click({detail: 1});
        const seen = timeline();
        for (const ms of [0, 100, 480, 900, 1200, 2500]) {
            assert.equal(seen[ms].text, NOTES[1]);
            assert.equal(seen[ms].count, '1');
            assert.equal(seen[ms].chat, 'open');
        }
        assert.equal(writes.length, 1);
    ''', reduced=True)


def test_tenth_leaf_shows_full_cup_at_once_and_disables_on_landing():
    _run('''
        leaf.events.click({detail: 1});
        assert.equal(current, 'Your cup is full. I’ll keep it warm while you read.');
        assert.equal(leaf.disabled, false, 'the last leaf stays visible while it falls');
        const seen = timeline();
        assert.equal(leaf.disabled, true);
        assert.equal(seen[2500].count, '10');
        assert.equal(writes.length, 1);
        leaf.events.click({detail: 1});
        assert.equal(saved['dailydigest-tea-leaves:day'], '10');
    ''', leaves=9)
