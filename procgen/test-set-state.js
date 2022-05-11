
const combos = [
	["ArrowLeft", "ArrowDown"],
	["ArrowLeft",],
	["ArrowLeft", "ArrowUp"],
	["ArrowDown",],
	[],
	["ArrowUp",],
	["ArrowRight", "ArrowDown"],
	["ArrowRight",],
	["ArrowRight", "ArrowUp"],
	["KeyD",],
	["KeyA",],
	["KeyW",],
	["KeyS",],
	["KeyQ",],
	["KeyE",],
];

let keyState = new Set();
function resetKeys() {
	keyState.clear();
}
function listenForKeys() {
	document.body.addEventListener("keydown", (e) => {
		keyState.add(e.code);
		e.preventDefault();
	});
}
function getAction() {
	let longest = -1;
	let action = -1;
	let i = 0;
	for (const combo of combos) {
		let hit = true;
		for (const k of combo) {
			if (!keyState.has(k)) {
				hit = false;
				break;
			}
		}
		if (hit && longest < combo.length) {
			longest = combo.length;
			action = i;
		}
		i++;
	}
	return action;
}

function printState(state, stats, screens, realtime) {
	if (!realtime) {
		const rgb = state.rgb;
		screens.appendChild(rgb);
	}
	delete state.rgb;
	let statsText = "";
	for (const [k, v] of Object.entries(state)) {
		statsText += String(k) + ": " + String(v) + "\n";
	}
	stats.innerText = statsText;
}

function parseOpts() {
	try {
		const search = new URLSearchParams(window.location.search);
		const ret = {};
		for (const [k, v] of search.entries()) {
			ret[k] = JSON.parse(v);
		}
		return ret;
	} catch (e) {
		console.error("Query string is invalid");
		return {};
	}
}

function deepcopy(obj) {
	return JSON.parse(JSON.stringify(obj));
}

async function main() {
	const div = document.getElementById("app");
	let opts = parseOpts();
	let realtime = false;
	if (opts.realtime !== undefined) {
		realtime = opts.realtime;
		delete opts.realtime;
	}
	const c = await CheerpGame.init({
		...CheerpGame.defaultOpts(),
		...opts,
	});
	div.appendChild(c.getCanvas());
	const stats = document.createElement("div");
	div.appendChild(stats);
	const screens = document.createElement("div");
	div.appendChild(screens);
	screens.style.overflowX = "auto";
	screens.style.whiteSpace = "nowrap";

	listenForKeys();

	c.render();
	const state = c.observe();
	const start_pos = [deepcopy(state.agent_x), deepcopy(state.agent_y)];
	printState(state, stats, screens, realtime);
	setInterval((start_pos) => {
		if (!realtime && keyState.size == 0)
			return;
		const action = getAction();
		c.step(action);
		c.render();
		const state = c.observe();
		printState(state, stats, screens, realtime);
		resetKeys();

		new_state = deepcopy(state);
		new_state.agent_x = start_pos[0];
		new_state.agent_y = start_pos[1];
		new_grid = new Int32Array();
		for (const [key, value] in Object.entries(new_state.grid)) {
			new_grid[key] = value;
		}
		new_grid[0] = 100;
		new_state.grid = new_grid;
		c.setState(new_state);
		c.render();
	}, 1000 / 15, start_pos);
}
