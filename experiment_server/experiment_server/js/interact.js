import { getAction } from './utils';

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
    const game = await CheerpGame.init({ // eslint-disable-line no-undef
        ...CheerpGame.defaultOpts(), // eslint-disable-line no-undef
        ...opts,
    });
    div.appendChild(game.getCanvas());
    listenForKeys();

    game.render();

    setInterval(() => {
        if (!realtime && keyState.size === 0) return;

        const action = getAction(keyState);
        game.step(action);
        game.render();

        resetKeys();
    }, 1000 / 15);
}

main();
