class Stats {
    constructor(div, document, realtime) {
        this.document = document;

        this.stats = this.document.createElement('div');
        div.appendChild(this.stats);

        this.realtime = realtime;
        if (!realtime) {
            this.screens = this.document.createElement('div');
            div.appendChild(this.screens);

            screens.style.overflowX = 'auto';
            screens.style.whiteSpace = 'nowrap';
        }
    }

    print(state) {
        if (!this.realtime) {
            const { rgb } = state;
            this.screens.append(rgb);
        }
        let statsText = '';
        for (const [k, v] of Object.entries(state)) { // eslint-disable-line no-restricted-syntax
            if (String(k) !== 'rgb') {
                statsText += `${String(k)}: ${String(v)}\n`;
            }
        }
        return [statsText, rgb];
    }
}