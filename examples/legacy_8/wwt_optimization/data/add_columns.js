const fs = require('fs');
const readline = require('readline');

const inputPath = process.argv[2] || 'WWTP_PO4_NH4_removal_backup.csv';
const outputPath = process.argv[3] || 'WWTP_PO4_NH4_removal.csv';

const STEP_MINUTES = 2;

const COL = {
    IRON_Input: 3,
    OXYGEN_input: 4,
    POLYALUMINUM_Input: 5,
    INLET_valve: 6,
    OUTLET_valve: 7
};

const chemicals = ['IRON_Input', 'OXYGEN_input', 'POLYALUMINUM_Input'];
const valves = ['INLET_valve', 'OUTLET_valve'];
const decayCols = [...chemicals, ...valves]; // 5 columns total

function computeRow(fields, chemState, valveState) {
    const chemValues = [];
    for (const name of chemicals) {
        const value = parseFloat(fields[COL[name]]);
        const st = chemState[name];

        if (value > 0) {
            st.minutesSinceLast = 0;
            st.lastBatchValue = value;
            chemValues.push(1 / value);
        } else if (st.lastBatchValue !== null) {
            st.minutesSinceLast += STEP_MINUTES;
            chemValues.push(st.minutesSinceLast / st.lastBatchValue);
        } else {
            chemValues.push(0);
        }
    }

    const valveValues = [];
    for (const name of valves) {
        const value = parseFloat(fields[COL[name]]);
        const st = valveState[name];

        if (st.currentState === null || value !== st.currentState) {
            st.currentState = value;
            st.minutesInState = 0;
        } else {
            st.minutesInState += STEP_MINUTES;
        }
        valveValues.push(st.minutesInState);
    }

    return [...chemValues, ...valveValues];
}

function freshState() {
    const chemState = {};
    for (const name of chemicals)
        chemState[name] = { minutesSinceLast: null, lastBatchValue: null };
    const valveState = {};
    for (const name of valves)
        valveState[name] = { currentState: null, minutesInState: 0 };
    return { chemState, valveState };
}

// --- Pass 1: compute all raw values, find min/max ---
console.log('Pass 1: computing raw values and min/max...');

const allRawValues = [];
const mins = new Array(5).fill(Infinity);
const maxs = new Array(5).fill(-Infinity);

const { chemState: cs1, valveState: vs1 } = freshState();
const rl1 = readline.createInterface({ input: fs.createReadStream(inputPath) });
let lineNum1 = 0;

rl1.on('line', (line) => {
    lineNum1++;
    if (lineNum1 === 1) return;

    const fields = line.split(',');
    const raw = computeRow(fields, cs1, vs1);
    allRawValues.push(raw);

    for (let i = 0; i < 5; i++) {
        if (raw[i] < mins[i]) mins[i] = raw[i];
        if (raw[i] > maxs[i]) maxs[i] = raw[i];
    }
});

rl1.on('close', () => {
    console.log('Mins:', mins);
    console.log('Maxs:', maxs);
    console.log(`Rows: ${allRawValues.length}`);

    // --- Pass 2: write output with normalized decay/duration columns ---
    console.log('Pass 2: writing normalized output...');

    const rl2 = readline.createInterface({ input: fs.createReadStream(inputPath) });
    const out = fs.createWriteStream(outputPath);
    let lineNum2 = 0;
    let dataRow = 0;

    rl2.on('line', (line) => {
        lineNum2++;

        if (lineNum2 === 1) {
            const headers = line.split(',');
            const newHeaders = [
                ...headers.slice(0, 8),
                'IRON_Input_decay',
                'OXYGEN_input_decay',
                'POLYALUMINUM_Input_decay',
                'INLET_valve_duration',
                'OUTLET_valve_duration',
                ...headers.slice(8)
            ];
            out.write(newHeaders.join(',') + '\n');
            return;
        }

        const fields = line.split(',');
        const raw = allRawValues[dataRow];
        dataRow++;

        const normalized = raw.map((v, i) => {
            const range = maxs[i] - mins[i];
            return range > 0 ? (v - mins[i]) / range : 0;
        });

        const newFields = [
            ...fields.slice(0, 8),
            ...normalized,
            ...fields.slice(8)
        ];
        out.write(newFields.join(',') + '\n');
    });

    rl2.on('close', () => {
        out.end(() => {
            console.log(`Done. Processed ${dataRow} data rows.`);
            console.log(`Output: ${outputPath}`);
        });
    });
});
