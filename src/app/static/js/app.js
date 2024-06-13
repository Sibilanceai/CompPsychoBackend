// document.addEventListener('DOMContentLoaded', function () {
//     const cy = cytoscape({
//         container: document.getElementById('cy'),
//         style: [
//             {
//                 selector: 'node',
//                 style: {
//                     'background-color': '#666',
//                     'label': 'data(id)'
//                 }
//             },
//             {
//                 selector: 'edge',
//                 style: {
//                     'width': 'mapData(absWeight, 0, 10, 2, 6)',  // Use absWeight for width mapping
//                     'line-color': 'mapData(absWeight, 0, 10, blue, red)', // Use absWeight for color gradient
//                     'target-arrow-color': '#ccc',
//                     'target-arrow-shape': 'triangle',
//                     'curve-style': 'bezier'
//                 }
//             }
//         ],
//         layout: {
//             name: 'grid'
//         }
//     });

//     let currentTimestep = 0;
//     let totalTimesteps = 0;

//     // Fetch the total number of timesteps when the page loads
//     fetch('http://127.0.0.1:5000/get_total_timesteps')
//         .then(response => response.json())
//         .then(data => {
//             totalTimesteps = data.totalTimesteps;
//             console.log(`Total timesteps available: ${totalTimesteps}`);
//         })
//         .catch(error => console.error('Error fetching total timesteps:', error));

//     function updateGraph() {
//         if (currentTimestep < totalTimesteps) {
//             fetch(`http://127.0.0.1:5000/get_graph/${currentTimestep}`)
//                 .then(response => {
//                     if (!response.ok) throw new Error('Failed to fetch graph data');
//                     return response.json();
//                 })
//                 .then(data => {
//                     // Modify edge data to include absolute weight
//                     console.log("Received graph data:", data);
//                     data.elements.edges.forEach(edge => {
//                         edge.data.absWeight = Math.abs(edge.data.weight);  // Directly modify each edge's data
//                     });
//                     console.log("Modified elements for visualization:", data.elements);

//                     cy.elements().remove();  // Remove the existing elements
//                     cy.add(data.elements);   // Add new elements from the fetched data with absWeight
//                     cy.layout({ name: 'grid' }).run();  // Re-run the layout
//                     console.log(`Updated graph for timestep ${currentTimestep}`);
//                     currentTimestep++;  // Increment to the next timestep
//                 })
//                 .catch(error => console.error('Error:', error));
//         } else {
//             console.log("No more timesteps available.");
//         }
//     }

//     // Event listener for the 'Next' button
//     document.getElementById('next').addEventListener('click', function() {
//         console.log("Stepping to the next graph");
//         updateGraph();
//     });
// });

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
                    'width': 'mapData(absWeight, 0, 10, 1, 10)', // Adjust the max weight and width accordingly
                    'line-color': 'mapData(absWeight, 0, 10, green, yellow)', // Adjust the color mapping
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'grid'
        }
    });

    let currentTimestep = 0;
    let totalTimesteps = 0;

    // Fetch the total number of timesteps when the page loads
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
                    console.log("Received graph data:", data);
                    const updatedElements = data.elements.edges.map(edge => ({
                        group: 'edges',
                        data: {
                            ...edge.data,
                            absWeight: Math.abs(edge.data.weight) // Calculate the absolute weight for visualization
                        }
                    }));

                    cy.elements().remove(); // Remove the existing elements
                    cy.add([...data.elements.nodes, ...updatedElements]); // Add nodes and updated edges to the graph
                    cy.layout({ name: 'grid' }).run(); // Re-run the layout
                    console.log(`Updated graph for timestep ${currentTimestep}`);
                    currentTimestep++; // Increment to the next timestep
                })
                .catch(error => console.error('Error:', error));
        } else {
            console.log("No more timesteps available.");
        }
    }

    // Event listener for the 'Next' button
    document.getElementById('next').addEventListener('click', function () {
        console.log("Stepping to the next graph");
        updateGraph();
    });
});