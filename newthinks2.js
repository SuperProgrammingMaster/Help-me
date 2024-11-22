

class NeuralNetwork {
    constructor(layers) {
        this.layers = layers; // 네트워크의 각 층 크기 배열
        this.nodes = []; // 각 노드들
        this.initNodes();
        this.epsilon = 1e-8;
    }

    // Sigmoid 함수와 그 도함수
    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    sigmoidDerivative(x) {
        let sig = this.sigmoid(x);
        return sig * (1 - sig);
    }

    // ReLU 함수와 그 도함수
    ReLU(x) {
        return Math.max(0, x);
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    heInitialization(size) {
        return Math.random() * Math.sqrt(2.0 / size);  // fan_in 값에 비례
    }
    // 노드 생성
    initNodes() {
        for (let i = 0; i < this.layers.length; i++) {
            this.nodes[i] = [];
            for (let k = 0; k < this.layers[i]; k++) {
                let temp = [];
                if (i > 0) {
                    for (let m = 0; m < this.layers[i - 1]; m++) {
                        temp.push(this.heInitialization(this.layers[i-1]));
                    }
                }
                this.nodes[i][k] = { w: temp, b: 0 };
            }
        }
    }

    // 모델 결과 계산
    getModelResult(x) {
        let layerOutputs = [];
        for (let l = 0; l < this.nodes.length; l++) {
            let temp = [];
            for (let k = 0; k < this.nodes[l].length; k++) {
                if (l == 0) {
                    temp.push(x);
                } else {
                    let sum = 0;
                    for (let j = 0; j < this.nodes[l][k].w.length; j++) {
                        sum += this.ReLU(layerOutputs[l - 1][j] * this.nodes[l][k].w[j] + this.nodes[l][k].b);
                    }
                    temp.push(sum);
                }
            }
            layerOutputs.push(temp);
        }
        return { output: layerOutputs[this.nodes.length - 1][0], layerOutputs: layerOutputs };
    }

    // Loss 계산
    loss(x) {
        return (datasets[x].y - this.getModelResult(x).output);
    }

    // Backpropagation을 통한 가중치 업데이트
    SGD(datasets, batchsize, maxepoch, learningRate, earlyexit = false) {
        if (batchsize > datasets.length) {
            batchsize = datasets.length;
        }

        for (let batch = 0; batch < batchsize; batch++) {
            for (let epoch = 0; epoch < maxepoch; epoch++) {
                let x = datasets[batch].x;
                let y = datasets[batch].y;

                let { layerOutputs, output } = this.getModelResult(x);


                if (earlyexit) {
                    if (Math.abs(y - output) > 5) {
                        return;
                    }
                }

                
                for (let l = this.nodes.length - 1; l >= 0; l--) {
                    for (let k = 0; k < this.nodes[l].length; k++) {
                        let delta = 0;
                        if (l == this.nodes.length - 1) {
                            delta = (this.ReLU(layerOutputs[l][k]) - y) * this.reluDerivative(layerOutputs[l][k]);
                        } else {
                            let m = this.nodes[l + 1].length;
                            for (let i = 0; i < m; i++) {
                                delta += this.nodes[l + 1][i].d * this.nodes[l + 1][i].w[k] * this.reluDerivative(layerOutputs[l][k]);
                            }
                        }
                        this.nodes[l][k] = { w: this.nodes[l][k].w, b: this.nodes[l][k].b, d: delta };

                        // 가중치와 편향 업데이트
                        for (let j = 0; j < this.nodes[l][k].w.length; j++) {
                            this.nodes[l][k].w[j] -= learningRate * this.nodes[l][k].d * this.ReLU(layerOutputs[l - 1][j]);
                        }
                        this.nodes[l][k].b -= learningRate * this.nodes[l][k].d;
                    }
                }
            }
        }
    }
    Adam(datasets, batchsize, maxepoch, learningRate, earlyexit = false,betas = [0.9,0.999])
    {
        if (batchsize > datasets.length) {
            batchsize = datasets.length;
        }
        let ms = [];
        let vs = [];

        for (let l = 0; l < this.nodes.length; l++) {
            ms[l] = [];
            vs[l] = [];

            for (let k = 0; k < this.nodes[l].length; k++) {
                ms[l][k] = new Array(this.nodes[l][k].w.length).fill(0);  // ms 초기화
                vs[l][k] = new Array(this.nodes[l][k].w.length).fill(0);  // vs 초기화
            }
        }
        for (let batch = 0; batch < batchsize; batch++) {
            
            for (let epoch = 0; epoch < maxepoch; epoch++) {
                let x = datasets[batch].x;
                let y = datasets[batch].y;

                let { layerOutputs, output } = this.getModelResult(x);


                if (earlyexit) {
                    if (Math.abs(y - output) > 5) {
                        return;
                    }
                }

                
                for (let l = this.nodes.length - 1; l >= 0; l--) {
                    for (let k = 0; k < this.nodes[l].length; k++) {
                        let delta = 0;
                        if (l == this.nodes.length - 1) {
                            delta = (this.ReLU(layerOutputs[l][k]) - y) * this.reluDerivative(layerOutputs[l][k]);
                        } else {
                            for (let i = 0; i < this.nodes[l + 1].length; i++) {
                                delta += this.nodes[l + 1][i].d * this.nodes[l + 1][i].w[k] * this.reluDerivative(layerOutputs[l][k]);
                            }
                        }
                        
                        this.nodes[l][k] = { w: this.nodes[l][k].w, b: this.nodes[l][k].b, d: delta};
                        
                        for (let j = 0; j < this.nodes[l][k].w.length; j++) {
                            
                            let g =this.nodes[l][k].d * this.ReLU(layerOutputs[l - 1][j])
                            let m = betas[0] * ms[l][k][j] + (1-betas[0]) * g
                            let v = betas[1] * vs[l][k][j] + (1-betas[1]) * g**2
                            let hatm = m/(1-betas[0])
                            let hatv = v/(1-betas[1])
                            this.nodes[l][k].w[j] -= (learningRate * hatm)/(Math.sqrt(hatv)+this.epsilon);
                            //console.log(this.nodes[l][k].w[j])
                        }
                        this.nodes[l][k].b -= learningRate * this.nodes[l][k].d;
                    }
                }
            }
        }
    }
}
let datasets = [];
let step = 0.01;

for (let i = 0; i <= 1000; i++) {
    let x = i * step; 
    let y = x ** 2 + 3; 
    datasets.push({ x, y });
}


let nn = new NeuralNetwork([1, 10, 10,10, 1]);


nn.Adam(datasets,1000,5000,0.001,false)



const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const fs = require('fs');

const width = 800;
const height = 600;
const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height });

let x = []
let y= []
for(const dataset of datasets)
{
    x.push(dataset.x)
    console.log(`x : ${dataset.x} y : ${nn.getModelResult(dataset.x).output}`)
    y.push(dataset.y - nn.getModelResult(dataset.x).output)
}

const createChart = async () => {
    const configuration = {
        type: 'line', 
        data: {
            labels: x,
            datasets: [
                {
                    label: '오차',
                    data: y, 
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)', 
                    fill: false, 
                    tension: 0.4, 
                },
            ],
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'x와 오차(y) 그래프',
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'x',
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: '오차 (y)',
                    },
                },
            },
        },
    };

    const image = await chartJSNodeCanvas.renderToBuffer(configuration);
    fs.writeFileSync('graph.png', image); 
    console.log('그래프가 graph.png로 저장되었습니다!');
};

createChart();











/*
const { exit } = require('process');
// 사용자 입력 처리
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});









// 사용자에게 입력 요청
//console.log("x의 값을 입력하시오");
/*
function start() {
    rl.question('', (input) => {
        if (input === "stop") {
            rl.close();
            return;
        } else {
            console.log(`정답 : ${nn.getModelResult(parseInt(input)).output}`);
            start();
        }
    });
}

start();
*/