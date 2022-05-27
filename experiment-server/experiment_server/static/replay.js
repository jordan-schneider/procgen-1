
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

async function request_question() {
    return await fetch("/question?env=miner&left_length=10&left_type=traj&right_length=10&right_type=traj", {
        method: 'GET',
        cache: 'no-store',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(res => res.json());
}

function prepare_state(state) {
    let new_grid = new Int32Array(state.grid.length);
    for (const [key, value] of Object.entries(state.grid)) {
        new_grid[parseInt(key)] = value;
    }

    let out = new Object();
    out.grid = new_grid;
    out.grid_width = state.grid_width;
    out.grid_height = state.grid_height;
    out.agent_x = state.agent_pos[0];
    out.agent_y = state.agent_pos[1];
    out.exit_x = state.exit_pos[0];
    out.exit_y = state.exit_pos[1];
    return out
}

function zip(a, b) {
    return a.map((k, i) => [k, b[i]]);
}

function make_game(game, traj) {
    return {
        game: game,
        traj: traj,
        playState: "paused",
        time: 0,
    }
}

let games = [];
let question = null;

async function main() {
    let opts = parseOpts();
    let realtime = false;
    if (opts.realtime !== undefined) {
        realtime = opts.realtime;
        delete opts.realtime;
    }
    const left_game = await CheerpGame.init({
        ...CheerpGame.defaultOpts(),
        ...opts,
    });
    const right_game = await CheerpGame.init({
        ...CheerpGame.defaultOpts(),
        ...opts,
    });
    games = [left_game, right_game];
    document.getElementById("leftGame").appendChild(left_game.getCanvas());
    document.getElementById("rightGame").appendChild(right_game.getCanvas());

    question = JSON.parse(await request_question());
    const left_traj = question.first_traj
    const right_traj = question.second_traj
    left_traj.start_state = prepare_state(left_traj.start_state);
    right_traj.start_state = prepare_state(right_traj.start_state);

    left_game.setState(left_traj.start_state);
    left_game.render();
    right_game.setState(right_traj.start_state);
    right_game.render();

    games = [make_game(left_game, left_traj), make_game(right_game, right_traj)];

    setInterval(() => {
        if (games[0].playState !== "paused") {
            const time = games[0].time;
            if (time < games[0].traj.actions.length) {
                const left_action = games[0].traj.actions[time];
                games[0].game.step(left_action);
                games[0].game.render();
                games[0].time += 1;
            }
        }
        if (games[1].playState !== "paused") {
            const time = games[1].time;
            if (time < games[1].traj.actions.length) {
                const right_action = games[1].traj.actions[games[1].time];
                games[1].game.step(right_action);
                games[1].game.render();
                games[1].time += 1;
            }
        }
    }, 1000);
}

async function pauseLeft() {
    games[0].playState = "paused";
}
async function playLeft() {
    games[0].playState = "playing";
}
async function restartLeft() {
    games[0].time = 0;
    games[0].playState = "paused";
    games[0].game.setState(games[0].traj.start_state);
    games[0].game.render();
}
async function pauseRight() {
    games[1].playState = "paused";
}
async function playRight() {
    games[1].playState = "playing";
}
async function restartRight() {
    games[1].time = 0;
    games[1].playState = "paused";
    games[1].game.setState(games[1].traj.start_state);
    games[1].game.render();
}

async function selectLeft() {
    fetch("/submit", {
        method: 'POST',
        cache: 'no-store',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            id: question.id,
            answer: "left",
        })
    });
}
async function selectRight() {
    fetch("/submit", {
        method: 'POST',
        cache: 'no-store',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            id: question.id,
            answer: "right",
        })
    });
}