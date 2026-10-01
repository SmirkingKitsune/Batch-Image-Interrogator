// A small global store: one state object, shallow updates, selector hooks.
import { useEffect, useRef, useState } from './lib.js';

const subscribers = new Set();
let state = {};
let scheduled = false;
let frame = 0;
let timer = 0;

// Longest a state change waits for a frame before subscribers run anyway.
const FLUSH_FALLBACK_MS = 100;

export function getState() {
  return state;
}

function flush() {
  if (!scheduled) return;
  scheduled = false;
  cancelAnimationFrame(frame);
  clearTimeout(timer);
  for (const fn of subscribers) {
    try {
      fn(state);
    } catch (err) {
      console.error('store subscriber failed', err);
    }
  }
}

/** Merge a partial update (or the result of fn(state)) into the state. */
export function setState(update) {
  const partial = typeof update === 'function' ? update(state) : update;
  if (!partial) return;
  state = { ...state, ...partial };
  if (!scheduled) {
    scheduled = true;
    // Coalesce bursts of backend events into one render per frame. Frames can
    // stall for most of a second while CUDA work (a model load or unload)
    // starves the compositor on a shared GPU, or while the window is hidden;
    // the timer keeps the DOM, and so button states and clicks, current.
    frame = requestAnimationFrame(flush);
    timer = setTimeout(flush, FLUSH_FALLBACK_MS);
  }
}

/** Update one nested slice: patch('interrog', { running: true }). */
export function patch(slice, update) {
  setState((s) => {
    const current = s[slice] || {};
    const partial = typeof update === 'function' ? update(current, s) : update;
    if (!partial) return null;
    return { [slice]: { ...current, ...partial } };
  });
}

export function initState(initial) {
  state = initial;
}

function shallowEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!Object.is(a[k], b[k])) return false;
  return true;
}

/** Re-render when the selected value changes (shallow comparison). */
export function useStore(selector) {
  const selectorRef = useRef(selector);
  selectorRef.current = selector;
  const [value, setValue] = useState(() => selector(state));
  const valueRef = useRef(value);
  valueRef.current = value;
  useEffect(() => {
    const listener = (next) => {
      const selected = selectorRef.current(next);
      if (!shallowEqual(selected, valueRef.current)) {
        valueRef.current = selected;
        setValue(() => selected);
      }
    };
    subscribers.add(listener);
    listener(state);
    return () => subscribers.delete(listener);
  }, []);
  return value;
}
