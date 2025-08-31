// Main Tetris App - Refactored for better organization
class TetrisApp {
    constructor() {
        // Initialize modules
        this.websocketManager = new WebSocketManager();
        this.statsManager = new StatsManager();
        this.renderer = new GameRenderer('gameCanvas');
        this.logger = new ActivityLogger('activityLog');
        
        // Legacy WebSocket references for compatibility
        this.websocket = null;
        this.playOnceWebSocket = null;
        this.playMultipleWebSocket = null;
        this.quickTrainWebSocket = null;
        
        // App state
        this.isStreaming = false;
        this.currentGame = null;
        this.debugMode = true;
        
        // Initialize
        this.initializeElements();
        this.statsManager.setElements(this.elements);
        this.setupEventListeners();
        this.checkHealth();
        this.updateTrainingControls();
        this.updateButtonStates();
        this.renderer.drawEmptyBoard();
        this.initializeQuickTrainingToggle();
    }

    // Proxy active operation to websocket manager
    get activeOperation() {
        return this.websocketManager.getActiveOperation();
    }

    set activeOperation(value) {
        if (value) {
            this.websocketManager.setActiveOperation(value);
        } else {
            this.websocketManager.clearActiveOperation();
        }
    }

    initializeElements() {
        this.elements = {
            // Buttons
            streamBtn: document.getElementById('streamBtn'),
            playOnceBtn: document.getElementById('playOnceBtn'),
            playMultipleBtn: document.getElementById('playMultipleBtn'),
            quickTrainBtn: document.getElementById('quickTrainBtn'),
            clearResultsBtn: document.getElementById('clearResultsBtn'),
            
            // Stats
            currentScore: document.getElementById('currentScore'),
            currentLines: document.getElementById('currentLines'),
            currentSteps: document.getElementById('currentSteps'),
            gameStatus: document.getElementById('gameStatus'),
            
            // Results
            totalEpisodes: document.getElementById('totalEpisodes'),
            avgScore: document.getElementById('avgScore'),
            totalLines: document.getElementById('totalLines'),
            bestScore: document.getElementById('bestScore'),
            maxScore: document.getElementById('maxScore'),
            
            // Controls
            episodes: document.getElementById('episodes'),
            seed: document.getElementById('seed'),
            algorithm: document.getElementById('algorithm'),
            
            // Training
            trainAlgo: document.getElementById('trainAlgo'),
            trainSeed: document.getElementById('trainSeed'),
            generations: document.getElementById('generations'),
            populationSize: document.getElementById('populationSize'),
            trainEpisodes: document.getElementById('trainEpisodes'),
            learningRate: document.getElementById('learningRate'),
            
            // Quick Training Toggle
            quickTrainingToggle: document.getElementById('quickTrainingToggle'),
            toggleStatus: document.getElementById('toggleStatus'),
            
            // Activity Log
            activityLog: document.getElementById('activityLog')
        };
        
        this.validateElements();
    }

    validateElements() {
        console.log('DOM Elements Status:');
        const criticalElements = ['currentScore', 'currentLines', 'avgScore', 'totalLines', 'bestScore'];
        for (const key of criticalElements) {
            const element = this.elements[key];
            if (!element) {
                console.error(`❌ MISSING: ${key}`);
            } else {
                console.log(`✅ FOUND: ${key} = ${element.textContent}`);
            }
        }
    }

    setupEventListeners() {
        // Button event listeners
        this.elements.streamBtn.addEventListener('click', () => this.toggleStream());
        this.elements.playOnceBtn.addEventListener('click', () => this.togglePlayOnce());
        this.elements.playMultipleBtn.addEventListener('click', () => this.togglePlayMultiple());
        this.elements.quickTrainBtn.addEventListener('click', () => this.toggleQuickTrain());
        this.elements.clearResultsBtn.addEventListener('click', () => this.resetStats());
        
        // Quick Training Toggle
        this.elements.quickTrainingToggle.addEventListener('change', (e) => 
            this.toggleQuickTraining(e.target.checked));
        
        // Algorithm change listeners
        this.elements.algorithm.addEventListener('change', () => this.updatePlayControls());
        this.elements.trainAlgo.addEventListener('change', () => this.updateTrainingControls());
        
        // Initialize controls
        this.updatePlayControls();
    }

    // Delegate stats methods
    resetStats() { this.statsManager.resetStats(); }
    addEpisode(score, lines) { this.statsManager.addEpisode(score, lines); }
    safeUpdateElement(id, value) { this.statsManager.safeUpdateElement(id, value); }

    // Delegate active operation methods  
    setActiveOperation(operation) {
        this.websocketManager.setActiveOperation(operation);
        this.updateButtonStates();
        this.logger.logWebSocket(`Started ${AlgorithmUtils.getAlgorithmDisplayName(operation)}`);
    }

    clearActiveOperation() {
        if (this.websocketManager.clearActiveOperation()) {
            this.updateButtonStates();
            this.logger.logWebSocket('Operation completed - buttons re-enabled');
            return true;
        }
        return false;
    }

    // Quick Training Toggle Management
    initializeQuickTrainingToggle() {
        const isEnabled = this.elements.quickTrainingToggle?.checked ?? true;
        this.toggleQuickTraining(isEnabled);
    }

    toggleQuickTraining(enabled) {
        const quickTrainBtn = this.elements.quickTrainBtn;
        const toggleStatus = this.elements.toggleStatus;
        
        if (enabled) {
            quickTrainBtn.classList.remove('hidden');
            quickTrainBtn.style.display = '';
            toggleStatus.textContent = 'ON';
            toggleStatus.classList.remove('off');
            this.logger.logSettings('Quick Training enabled - button now visible');
        } else {
            quickTrainBtn.classList.add('hidden');
            quickTrainBtn.style.display = 'none';
            toggleStatus.textContent = 'OFF';
            toggleStatus.classList.add('off');
            this.logger.logSettings('Quick Training disabled - button hidden');
            
            if (this.activeOperation === 'quickTrain') {
                this.logger.logWarning('Stopping active Quick Training due to toggle disable');
                this.toggleQuickTrain();
            }
        }
        
        this.updateButtonStates();
    }

    updateButtonStates() {
        const buttons = [
            { element: this.elements.streamBtn, operation: 'stream' },
            { element: this.elements.playOnceBtn, operation: 'playOnce' },
            { element: this.elements.playMultipleBtn, operation: 'playMultiple' },
            { element: this.elements.quickTrainBtn, operation: 'quickTrain' }
        ];
        
        buttons.forEach(({ element, operation }) => {
            if (!element) return;
            
            // Special handling for Quick Train button
            if (operation === 'quickTrain') {
                const toggleEnabled = this.elements.quickTrainingToggle?.checked;
                if (!toggleEnabled) {
                    element.classList.add('hidden');
                    element.style.display = 'none';
                    element.disabled = true;
                    return;
                } else {
                    element.classList.remove('hidden');
                    element.style.display = '';
                }
            }
            
            // Handle enabled/disabled state
            if (this.activeOperation === null) {
                element.disabled = false;
            } else if (this.activeOperation === operation) {
                element.disabled = false;
            } else {
                element.disabled = true;
            }
        });
    }

    // Health Check
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            this.logger.logSuccess('Health check passed', `Status: ${data.status}`);
        } catch (error) {
            this.logger.logError('Health check failed', error.message);
        }
    }

    // Control Updates
    updatePlayControls() {
        const algo = this.elements.algorithm.value;
        const sections = ['playGreedyControls', 'playTabuControls', 'playAnnealControls', 'playAcoControls'];
        
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.style.display = 'none';
            }
        });
        
        if (['greedy', 'tabu', 'anneal', 'aco'].includes(algo)) {
            const targetSection = document.getElementById(`play${algo.charAt(0).toUpperCase() + algo.slice(1)}Controls`);
            if (targetSection) {
                targetSection.style.display = 'block';
            }
        }
    }

    updateTrainingControls() {
        const algo = this.elements.trainAlgo.value;
        const cemControls = document.getElementById('cemControls');
        const reinforceControls = document.getElementById('reinforceControls');
        const otherControls = document.getElementById('otherAlgoControls');
        
        // Hide all first
        [cemControls, reinforceControls, otherControls].forEach(control => {
            if (control) control.style.display = 'none';
        });
        
        // Show relevant controls
        if (algo === 'cem' && cemControls) {
            cemControls.style.display = 'block';
        } else if (algo === 'reinforce' && reinforceControls) {
            reinforceControls.style.display = 'block';
        } else if (['greedy', 'tabu', 'anneal', 'aco'].includes(algo) && otherControls) {
            otherControls.style.display = 'block';
            this.updateOtherAlgoControls(algo);
        }
    }

    updateOtherAlgoControls(algo) {
        const sections = ['greedyControls', 'tabuControls', 'annealControls', 'acoControls'];
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.style.display = sectionId === `${algo}Controls` ? 'block' : 'none';
            }
        });
    }

    // Game Operations - Simplified
    async toggleStream() {
        if (this.isStreaming) {
            this.stopStream();
        } else {
            this.startStream();
        }
    }

    async togglePlayOnce() {
        if (this.playOnceWebSocket) {
            this.stopPlayOnce();
        } else {
            this.startPlayOnce();
        }
    }

    async togglePlayMultiple() {
        if (this.playMultipleWebSocket) {
            this.stopPlayMultiple();
        } else {
            this.startPlayMultiple();
        }
    }

    async toggleQuickTrain() {
        if (this.quickTrainWebSocket) {
            this.stopQuickTrain();
        } else {
            this.startQuickTrain();
        }
    }

    // Implementation methods would continue here...
    // For brevity, I'll create a separate file for the game operations
}
