document.addEventListener('DOMContentLoaded', function () {
    const cy = cytoscape({
        container: document.getElementById('cy'),
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#666',
                    'label': 'data(id)'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': function(edge) {
                        // Convert negative values to positive and apply a scaling factor
                        return 2 + 10 * Math.log10(Math.abs(edge.data('weight')) + 1); // Log scale with offset for visual distinction
                    },
                    'line-color': 'mapData(absWeight, 0, 10, blue, red)',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'cose',
            idealEdgeLength: function (edge) {
                return 100 / edge.data('absWeight');  // Adjust edge length as needed
            },
            nodeOverlap: 20,
            refresh: 20,
            fit: true,
            padding: 30,
            randomize: false,
            componentSpacing: 100,
            nodeRepulsion: 2048,
            edgeElasticity: function (edge) {
                return Math.max(1, edge.data('absWeight'));  // Use absolute weight for elasticity
            },
            nestingFactor: 5
        }
    });

    let currentTimestep = 0;
    let totalTimesteps = 0;

    fetch('http://127.0.0.1:5000/get_total_timesteps')
        .then(response => response.json())
        .then(data => {
            totalTimesteps = data.totalTimesteps;
            console.log(`Total timesteps available: ${totalTimesteps}`);
        })
        .catch(error => console.error('Error fetching total timesteps:', error));

    function updateGraph() {
        if (currentTimestep < totalTimesteps) {
            fetch(`http://127.0.0.1:5000/get_graph/${currentTimestep}`)
                .then(response => {
                    if (!response.ok) throw new Error('Failed to fetch graph data');
                    return response.json();
                })
                .then(data => {
                    data.elements.edges.forEach(edge => {
                        edge.data.absWeight = Math.abs(edge.data.weight);
                    });

                    cy.elements().remove();
                    cy.add(data.elements);
                    cy.layout({ name: 'cose' }).run();
                    console.log(`Updated graph for timestep ${currentTimestep}`);
                    currentTimestep++;
                })
                .catch(error => console.error('Error:', error));
        } else {
            console.log("No more timesteps available.");
        }
    }

    document.getElementById('next').addEventListener('click', function() {
        console.log("Stepping to the next graph");
        updateGraph();
    });
});