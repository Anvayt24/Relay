// Small helpers for building the page. Text always goes in as text, never as HTML: a model's
// answer is untrusted, and the page never uses innerHTML.

type Child = Node | string | null | undefined;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, value);
  for (const child of children) {
    if (child === null || child === undefined) continue;
    element.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return element;
}

export function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`missing #${id} in index.html`);
  return found as T;
}

export function clear(node: Element): void {
  node.replaceChildren();
}

export function table(headers: string[], rows: (Child[])[]): HTMLTableElement {
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, ...headers.map((name) => h("th", { scope: "col" }, name)))),
    h("tbody", {}, ...rows.map((row) => h("tr", {}, ...row.map((cell) => h("td", {}, cell))))),
  );
}

/** A term and its value, for a <dl>. */
export function term(label: string, value: Child, note?: string): Node[] {
  return [
    h("dt", {}, label),
    h("dd", {}, value, note ? h("span", { class: "note" }, ` ${note}`) : null),
  ];
}
