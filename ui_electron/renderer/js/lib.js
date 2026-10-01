// Preact + htm in one prebuilt module (node_modules/htm/preact/standalone),
// served by the bridge. No build step: templates are tagged literals.
export {
  html,
  render,
  h,
  Component,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from '/vendor/htm-preact.mjs';
