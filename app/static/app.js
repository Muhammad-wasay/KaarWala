const $ = (id) => document.getElementById(id);
const output = $("output");
let OPTIONS = null;

const SOURCE_TEXT = {
  exact: "Market data: exact match for this model and year.",
  model: "Market data: model-level average (no exact year match).",
  none: "Market data: no listings for this model, overall averages used.",
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtPkr(value) {
  return value >= 1e7
    ? "PKR " + (value / 1e7).toFixed(2) + " Cr"
    : "PKR " + (value / 1e5).toFixed(2) + " L";
}

function fillSelect(select, items, opts = {}) {
  select.innerHTML = "";
  if (opts.emptyLabel !== undefined) {
    const o = el("option", null, opts.emptyLabel);
    o.value = "";
    select.appendChild(o);
  }
  for (const item of items) {
    const o = el("option", null, item);
    o.value = item;
    select.appendChild(o);
  }
  if (opts.preferred && items.includes(opts.preferred)) select.value = opts.preferred;
}

function onBrandChange() {
  fillSelect($("model"), OPTIONS.models_by_brand[$("brand").value] || []);
  onModelChange();
}

function onModelChange() {
  const key = $("brand").value + "|" + $("model").value;
  fillSelect($("variant"), OPTIONS.variants[key] || [], { emptyLabel: "Not sure / not listed" });
}

function onFuelChange() {
  const electric = $("fuel").value === "Electric";
  $("engineField").classList.toggle("hidden", electric);
  $("engine").disabled = electric;
}

function renderPlaceholder() {
  output.innerHTML = "";
  const box = el("div", "placeholder");
  box.appendChild(el("div", "icon", "🔍"));
  box.appendChild(el("h3", null, "Ready when you are"));
  const p = el("div", null, "Fill in the details and press ");
  p.appendChild(el("b", null, "Estimate fair price"));
  p.appendChild(document.createTextNode("."));
  box.appendChild(p);
  output.appendChild(box);
}

function renderWarnings(list) {
  if (!list || list.length === 0) return;
  const box = el("div", "warnings");
  list.forEach((w) => box.appendChild(el("div", null, "⚠ " + w)));
  output.appendChild(box);
}

function renderErrors(errors, warnings) {
  output.innerHTML = "";
  const box = el("div", "errors");
  box.appendChild(el("strong", null, "Please fix the following:"));
  errors.forEach((e) => box.appendChild(el("div", null, "• " + e)));
  output.appendChild(box);
  renderWarnings(warnings);
}

function renderResult(d) {
  output.innerHTML = "";

  const chips = el("div", "chips");
  const mileageText = Number(d.car.mileage).toLocaleString() + " km";
  [d.car.brand + " " + d.car.model, String(d.car.year), mileageText,
   d.car.fuel_type, d.car.transmission, d.car.city]
    .forEach((t) => chips.appendChild(el("span", "chip", t)));
  output.appendChild(chips);

  output.appendChild(el("div", "label", "Estimated fair price"));
  output.appendChild(el("div", "price", fmtPkr(d.predicted_pkr)));

  output.appendChild(el("div", "label", "Likely range"));
  const span = d.range_high_pkr - d.range_low_pkr;
  let pos = span > 0 ? ((d.predicted_pkr - d.range_low_pkr) / span) * 100 : 50;
  pos = Math.min(Math.max(pos, 0), 100);
  const bar = el("div", "bar");
  const dot = el("div", "dot");
  dot.style.left = pos + "%";
  bar.appendChild(dot);
  output.appendChild(bar);

  const ends = el("div", "ends");
  ends.appendChild(el("span", null, fmtPkr(d.range_low_pkr)));
  ends.appendChild(el("span", null, fmtPkr(d.range_high_pkr)));
  output.appendChild(ends);

  if (d.verdict) {
    const v = d.verdict;
    const titles = { fair: "Looks fair", above: "Above market value", below: "Below market value" };
    const texts = {
      fair: "Asking price is within the typical error margin of the estimate.",
      above: "Asking is " + fmtPkr(Math.abs(v.difference_pkr)) + " (" + v.difference_pct + "%) above the estimate. Negotiate.",
      below: "Asking is " + fmtPkr(Math.abs(v.difference_pkr)) + " (" + Math.abs(v.difference_pct) + "%) below the estimate. Check the car's condition before buying.",
    };
    const box = el("div", "verdict " + v.status);
    box.appendChild(el("div", "vt", titles[v.status]));
    box.appendChild(el("div", null, texts[v.status]));
    output.appendChild(box);
  }

  output.appendChild(el("div", "src", SOURCE_TEXT[d.market_source] || ""));

  const notes = [...(d.warnings || [])];
  if (d.high_value) {
    notes.push("High-value car: the real error is usually much larger than the range shown.");
  }
  renderWarnings(notes);
}

async function loadOptions() {
  const res = await fetch("/options");
  OPTIONS = await res.json();

  fillSelect($("brand"), OPTIONS.brands);
  fillSelect($("fuel"), OPTIONS.fuel_types, { preferred: "Petrol" });
  fillSelect($("transmission"), OPTIONS.transmissions);
  fillSelect($("city"), OPTIONS.cities, { emptyLabel: "Choose a city" });

  const L = OPTIONS.limits;
  $("year").min = L.min_year; $("year").max = L.max_year; $("year").value = 2018;
  $("mileage").min = 0; $("mileage").max = L.max_mileage; $("mileage").value = 60000;
  $("engine").min = L.min_engine_cc; $("engine").max = L.max_engine_cc; $("engine").value = 1300;

  onBrandChange();
  onFuelChange();
}

$("brand").addEventListener("change", onBrandChange);
$("model").addEventListener("change", onModelChange);
$("fuel").addEventListener("change", onFuelChange);

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("submit");
  button.disabled = true;
  button.textContent = "Estimating…";

  const electric = $("fuel").value === "Electric";
  const askingRaw = $("asking").value.trim();
  const body = {
    brand: $("brand").value,
    model: $("model").value,
    variant: $("variant").value || null,
    year: parseInt($("year").value, 10),
    mileage_km: parseFloat($("mileage").value),
    fuel_type: $("fuel").value,
    transmission: $("transmission").value,
    engine_cc: electric ? null : parseFloat($("engine").value),
    city: $("city").value,
    asking_price_pkr: askingRaw === "" ? null : parseFloat(askingRaw),
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      renderErrors(data.errors || [data.detail || "Something went wrong."], data.warnings);
    } else {
      renderResult(data);
    }
  } catch (err) {
    renderErrors(["Could not reach the server. Is uvicorn running?"]);
  } finally {
    button.disabled = false;
    button.textContent = "Estimate fair price";
  }
});

renderPlaceholder();
loadOptions().catch(() => renderErrors(["Could not load dropdown options. Is the server running?"]));