import { Stats } from './stats';
import { getAction, post } from './utils';

let game = null;
let startState = null;
let actions = null;
let recording = false;
let trajIds = [];

const keyState = new Set();

function resetKeys() {
    keyState.clear();
}

function listenForKeys() {
    document.body.addEventListener('keydown', (e) => {
        keyState.add(e.code);
        e.preventDefault();
    });
}

function parseOpts() {
    try {
        const search = new URLSearchParams(window.location.search);
        const ret = {};
        for (const [k, v] of search.entries()) { // eslint-disable-line no-restricted-syntax
            ret[k] = JSON.parse(v);
        }
        return ret;
    } catch (e) {
        console.error('Query string is invalid');
        return {};
    }
}

async function main() {
    const div = document.getElementById('app');
    const opts = parseOpts();
    let realtime = false;
    if (opts.realtime !== undefined) {
        realtime = opts.realtime;
        delete opts.realtime;
    }
    game = await CheerpGame.init({ // eslint-disable-line no-undef
        ...CheerpGame.defaultOpts(), // eslint-disable-line no-undef
        ...opts,
    });
    div.appendChild(game.getCanvas());
    listenForKeys();

    game.render();
    let stats = null;
    if (opts.debug !== undefined) {
        stats = Stats(div, document);
        stats.print(game.observe());
    }

    setInterval(() => {
        if (!realtime && keyState.size === 0) return;

        const action = getAction(keyState);
        if (recording && action >= 0) {
            actions.push(action);
        }
        game.step(action);

        game.render();

        if (stats !== null) {
            stats.print(game.observe());
        }
        resetKeys();
    }, 1000 / 15);
}

async function startRecording() {
    recording = true;
    startState = game.getState().grid;
    actions = [];
}

async function submitQuestion() {
    if (trajIds.length === 2) {
        post('/submit_question', JSON.stringify({
            traj_ids: trajIds,
        }));
    }
}
async function submitRecording() {
    if (!recording || startState === null || actions === null) {
        return;
    }
    recording = false;

    await post('/record', JSON.stringify({
        start_state: startState,
        actions: actions,
    })).then(recordTrajectoryId);
    startState = null;
    actions = null;
    if (trajIds.length >= 2) {
        await submitQuestion();
        trajIds = [];
    }
}
async function recordTrajectoryId(response) {
    const data = await response.json();
    trajIds.push(data.trajectory_id);
}

async function cancelRecording() {
    recording = false;
    startState = null;
    actions = null;
}

window.startRecording = startRecording;
window.submitRecording = submitRecording;
window.cancelRecording = cancelRecording;

main();
