// Algorithm Parameters Utility
class AlgorithmUtils {
    static getAlgorithmDisplayName(algo) {
        const names = {
            'cem': 'CEM',
            'reinforce': 'REINFORCE',
            'greedy': 'Greedy (Nurse Dictator)',
            'tabu': 'Tabu Search (Nurse Gossip)',
            'anneal': 'Simulated Annealing (Coffee Break)',
            'aco': 'Ant Colony (Night Shift Ant March)'
        };
        return names[algo] || algo.toUpperCase();
    }

    static collectAlgorithmParams(algo) {
        const params = {};
        
        if (algo === 'greedy') {
            params.w_holes = parseFloat(document.getElementById('greedy_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('greedy_w_max_height')?.value || '1.0');
            params.w_bumpiness = parseFloat(document.getElementById('greedy_w_bumpiness')?.value || '1.0');
        } else if (algo === 'tabu') {
            params.tenure = parseInt(document.getElementById('tabu_tenure')?.value || '25');
            params.neighborhood_top_k = parseInt(document.getElementById('tabu_neighborhood_k')?.value || '10');
            params.aspiration = document.getElementById('tabu_aspiration')?.checked !== false;
            params.w_holes = parseFloat(document.getElementById('tabu_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('tabu_w_max_height')?.value || '1.0');
            params.rng_seed = parseInt(document.getElementById('trainSeed')?.value || '42');
        } else if (algo === 'anneal') {
            const T0_input = document.getElementById('anneal_T0')?.value;
            if (T0_input && T0_input.trim() !== '') {
                params.T0 = parseFloat(T0_input);
            }
            params.alpha = parseFloat(document.getElementById('anneal_alpha')?.value || '0.99');
            params.proposal_top_k = parseInt(document.getElementById('anneal_proposal_k')?.value || '10');
            params.w_holes = parseFloat(document.getElementById('anneal_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('anneal_w_max_height')?.value || '1.0');
            params.rng_seed = parseInt(document.getElementById('trainSeed')?.value || '42');
        } else if (algo === 'aco') {
            params.alpha = parseFloat(document.getElementById('aco_alpha')?.value || '1.0');
            params.beta = parseFloat(document.getElementById('aco_beta')?.value || '2.0');
            params.rho = parseFloat(document.getElementById('aco_rho')?.value || '0.1');
            params.num_ants = parseInt(document.getElementById('aco_ants')?.value || '20');
            params.w_holes = parseFloat(document.getElementById('aco_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('aco_w_max_height')?.value || '1.0');
            params.rng_seed = parseInt(document.getElementById('trainSeed')?.value || '42');
        }
        
        return params;
    }

    static collectPlayAlgorithmParams(algo) {
        const params = {};
        
        if (algo === 'greedy') {
            params.w_holes = parseFloat(document.getElementById('play_greedy_w_holes')?.value || '0.8');
            params.w_max_height = parseFloat(document.getElementById('play_greedy_w_max_height')?.value || '0.4');
            params.w_bumpiness = parseFloat(document.getElementById('play_greedy_w_bumpiness')?.value || '0.3');
        } else if (algo === 'tabu') {
            params.tenure = parseInt(document.getElementById('play_tabu_tenure')?.value || '7');
            params.neighborhood_top_k = parseInt(document.getElementById('play_tabu_neighborhood_k')?.value || '10');
            params.aspiration = document.getElementById('play_tabu_aspiration')?.checked !== false;
        } else if (algo === 'anneal') {
            const T0_input = document.getElementById('play_anneal_T0')?.value;
            if (T0_input && T0_input.trim() !== '') {
                params.T0 = parseFloat(T0_input);
            }
            params.alpha = parseFloat(document.getElementById('play_anneal_alpha')?.value || '0.99');
        } else if (algo === 'aco') {
            params.alpha = parseFloat(document.getElementById('play_aco_alpha')?.value || '1.0');
            params.beta = parseFloat(document.getElementById('play_aco_beta')?.value || '2.0');
            params.num_ants = parseInt(document.getElementById('play_aco_ants')?.value || '20');
        }
        
        return params;
    }
}
