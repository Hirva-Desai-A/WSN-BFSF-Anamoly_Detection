<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WSN IDS – ML Pipeline Flowchart</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap');

        *,
        *::before,
        *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background: #fff;
            font-family: sans-serif;
            color: #111;
            padding: 48px 20px 64px;
        }

        h1 {
            text-align: center;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .subtitle {
            text-align: center;
            font-family: 'Space Mono', monospace;
            font-size: 0.65rem;
            color: #777;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 48px;
        }

        .flow {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 820px;
            margin: 0 auto;
        }

        /* Standard node */
        .node {
            width: 100%;
            max-width: 480px;
            border: 2px solid #111;
            border-radius: 8px;
            padding: 14px 24px;
            text-align: center;
            animation: fadeUp 0.4s ease both;
        }

        .node.wide {
            max-width: 820px;
        }

        .node .step {
            font-family: 'Space Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 5px;
        }

        .node .label {
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.3;
        }

        /* Arrow */
        .arrow {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            max-width: 480px;
            animation: fadeUp 0.4s ease both;
        }

        .arrow.wide {
            max-width: 820px;
        }

        .arrow-line {
            width: 2px;
            height: 28px;
            background: #111;
        }

        .arrow-head {
            width: 0;
            height: 0;
            border-left: 7px solid transparent;
            border-right: 7px solid transparent;
            border-top: 10px solid #111;
        }

        /* Branch */
        .branch-row {
            display: flex;
            gap: 14px;
            width: 100%;
            max-width: 820px;
            animation: fadeUp 0.4s ease both;
        }

        .branch-col {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .branch-node {
            width: 100%;
            border: 2px solid #111;
            border-radius: 8px;
            padding: 13px 8px;
            text-align: center;
        }

        .branch-node .step {
            font-family: 'Space Mono', monospace;
            font-size: 0.54rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 5px;
        }

        .branch-node .label {
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .branch-node .stat {
            font-family: 'Space Mono', monospace;
            font-size: 0.65rem;
            color: #444;
            margin-top: 6px;
            font-weight: 700;
        }

        .branch-drops {
            display: flex;
            width: 100%;
            max-width: 820px;
            gap: 14px;
            animation: fadeUp 0.4s ease both;
        }

        .branch-drops .drop {
            flex: 1;
            display: flex;
            justify-content: center;
        }

        .branch-drops .drop .arrow-line {
            height: 22px;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .flow>* {
            animation-delay: calc(var(--i, 0) * 0.055s);
        }
    </style>
</head>

<body>

    <h1>ML-Based IDS — WSN Pipeline</h1>
    <p class="subtitle">Stacking Ensemble · End-to-End Flowchart</p>

    <div class="flow">

        <div class="node" style="--i:0">
            <div class="step">Step 1</div>
            <div class="label">Raw WSN Traffic Dataset</div>
        </div>

        <div class="arrow" style="--i:1">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node" style="--i:2">
            <div class="step">Step 2</div>
            <div class="label">Feature Selection (7 Features)</div>
        </div>

        <div class="arrow" style="--i:3">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node" style="--i:4">
            <div class="step">Step 3</div>
            <div class="label">Stratified Train / Test Split</div>
        </div>

        <div class="arrow" style="--i:5">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node" style="--i:6">
            <div class="step">Step 4</div>
            <div class="label">Standardisation + SMOTE</div>
        </div>

        <div class="arrow wide" style="--i:7">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node wide" style="--i:8">
            <div class="step">Step 5</div>
            <div class="label">Model Training — Four Classifiers in Parallel</div>
        </div>

        <div class="branch-drops" style="--i:9">
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
        </div>

        <div class="branch-row" style="--i:10">
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Baseline</div>
                    <div class="label">Logistic Regression</div>
                </div>
            </div>
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Base Learner A</div>
                    <div class="label">Random Forest</div>
                </div>
            </div>
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Base Learner B</div>
                    <div class="label">XGBoost</div>
                </div>
            </div>
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Proposed</div>
                    <div class="label">Stacking Ensemble</div>
                </div>
            </div>
        </div>

        <div class="branch-drops" style="--i:11">
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
        </div>

        <div class="node wide" style="--i:12">
            <div class="step">Step 6 — Stacking Architecture</div>
            <div class="label">5-Fold CV → Meta-Features → LR Meta-Learner</div>
        </div>

        <div class="arrow wide" style="--i:13">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node" style="--i:14">
            <div class="step">Step 7</div>
            <div class="label">Test Set Inference</div>
        </div>

        <div class="arrow" style="--i:15">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node" style="--i:16">
            <div class="step">Step 8</div>
            <div class="label">Evaluation — Accuracy · F1 · AUC · Confusion Matrix</div>
        </div>

        <div class="arrow wide" style="--i:17">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node wide" style="--i:18">
            <div class="step">Step 9</div>
            <div class="label">Model Performance Comparison</div>
        </div>

        <div class="branch-drops" style="--i:19">
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
            <div class="drop">
                <div class="arrow-line"></div>
            </div>
        </div>

        <div class="branch-row" style="--i:20">
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Logistic Regression</div>
                    <div class="label">82.54%</div>
                    <div class="stat">AUC 0.853</div>
                </div>
            </div>
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Random Forest</div>
                    <div class="label">99.51%</div>
                    <div class="stat">AUC 0.9997</div>
                </div>
            </div>
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">XGBoost</div>
                    <div class="label">99.59%</div>
                    <div class="stat">AUC 0.9998</div>
                </div>
            </div>
            <div class="branch-col">
                <div class="branch-node">
                    <div class="step">Stacking Ensemble ★</div>
                    <div class="label">99.61%</div>
                    <div class="stat">AUC 1.000</div>
                </div>
            </div>
        </div>

        <div class="arrow wide" style="--i:21">
            <div class="arrow-line"></div>
            <div class="arrow-head"></div>
        </div>

        <div class="node wide" style="--i:22">
            <div class="step">Step 10</div>
            <div class="label">Deployment-Ready IDS Alert System</div>
        </div>

    </div>

</body>

</html>