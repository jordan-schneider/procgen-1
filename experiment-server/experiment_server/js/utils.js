export function deepcopy(obj) {
    return JSON.parse(JSON.stringify(obj));
}

export function zip(a, b) {
    return a.map((k, i) => [k, b[i]]);
}

export function post(url, body) {
    return fetch(url, {
        method: 'POST',
        cache: 'no-store',
        headers: {
            'Content-Type': 'application/json',
        },
        body: body
    });
}

export const combos = [
    ['ArrowLeft', 'ArrowDown'],
    ['ArrowLeft'],
    ['ArrowLeft', 'ArrowUp'],
    ['ArrowDown'],
    [],
    ['ArrowUp'],
    ['ArrowRight', 'ArrowDown'],
    ['ArrowRight'],
    ['ArrowRight', 'ArrowUp'],
    ['KeyD'],
    ['KeyA'],
    ['KeyW'],
    ['KeyS'],
    ['KeyQ'],
    ['KeyE'],
];

export function getAction(keyState) {
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
        i += 1;
    }
    return action;
}