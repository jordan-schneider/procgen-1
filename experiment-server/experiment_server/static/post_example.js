async function main() {
    fetch("/submit", {
        method: 'POST',
        cache: 'no-store',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: "hello world" })
    });
}

