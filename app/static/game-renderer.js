// Game Rendering Module
class GameRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Create offscreen canvas for double buffering
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCanvas.width = this.canvas.width;
        this.offscreenCanvas.height = this.canvas.height;
        this.offscreenCtx = this.offscreenCanvas.getContext('2d');
        
        // Tetris piece colors
        this.tetrisPieceColors = [
            '#FF0D72', '#0DC2FF', '#0DFF72', '#FFB70D', '#B70DFF',
            '#FF720D', '#72FF0D', '#FF0D72', '#0D72FF'
        ];
        
        this.CELL_SIZE = 20;
    }

    drawEmptyBoard() {
        const width = this.canvas.width / this.CELL_SIZE;
        const height = this.canvas.height / this.CELL_SIZE;
        
        this.offscreenCtx.clearRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
        this.offscreenCtx.fillStyle = '#000';
        this.offscreenCtx.fillRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
        
        // Draw grid lines
        this.offscreenCtx.strokeStyle = '#333';
        this.offscreenCtx.lineWidth = 0.5;
        
        for (let x = 0; x <= width; x++) {
            this.offscreenCtx.beginPath();
            this.offscreenCtx.moveTo(x * this.CELL_SIZE, 0);
            this.offscreenCtx.lineTo(x * this.CELL_SIZE, this.offscreenCanvas.height);
            this.offscreenCtx.stroke();
        }
        
        for (let y = 0; y <= height; y++) {
            this.offscreenCtx.beginPath();
            this.offscreenCtx.moveTo(0, y * this.CELL_SIZE);
            this.offscreenCtx.lineTo(this.offscreenCanvas.width, y * this.CELL_SIZE);
            this.offscreenCtx.stroke();
        }
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.offscreenCanvas, 0, 0);
    }

    drawBoard(board, currentPiece = null) {
        if (!board || !Array.isArray(board)) {
            console.warn('Invalid board data received');
            return;
        }
        
        this.offscreenCtx.clearRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
        this.offscreenCtx.fillStyle = '#000';
        this.offscreenCtx.fillRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
        
        const rows = board.length;
        const cols = board[0]?.length || 10;
        
        // Draw placed pieces
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const cellValue = board[row][col];
                if (cellValue > 0) {
                    const colorIndex = (cellValue - 1) % this.tetrisPieceColors.length;
                    this.offscreenCtx.fillStyle = this.tetrisPieceColors[colorIndex];
                    this.offscreenCtx.fillRect(
                        col * this.CELL_SIZE + 1,
                        row * this.CELL_SIZE + 1,
                        this.CELL_SIZE - 2,
                        this.CELL_SIZE - 2
                    );
                }
            }
        }
        
        // Draw current falling piece if provided
        if (currentPiece) {
            this.drawCurrentPiece(currentPiece);
        }
        
        // Draw grid lines
        this.drawGridLines(rows, cols);
        
        // Copy to main canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.offscreenCanvas, 0, 0);
    }

    drawCurrentPiece(piece) {
        if (!piece || !piece.shape) return;
        
        const colorIndex = (piece.type - 1) % this.tetrisPieceColors.length;
        this.offscreenCtx.fillStyle = this.tetrisPieceColors[colorIndex];
        this.offscreenCtx.globalAlpha = 0.8;
        
        for (let row = 0; row < piece.shape.length; row++) {
            for (let col = 0; col < piece.shape[row].length; col++) {
                if (piece.shape[row][col]) {
                    const x = (piece.col + col) * this.CELL_SIZE;
                    const y = (piece.row + row) * this.CELL_SIZE;
                    this.offscreenCtx.fillRect(x + 1, y + 1, this.CELL_SIZE - 2, this.CELL_SIZE - 2);
                }
            }
        }
        
        this.offscreenCtx.globalAlpha = 1.0;
    }

    drawGridLines(rows, cols) {
        this.offscreenCtx.strokeStyle = '#333';
        this.offscreenCtx.lineWidth = 0.5;
        
        for (let x = 0; x <= cols; x++) {
            this.offscreenCtx.beginPath();
            this.offscreenCtx.moveTo(x * this.CELL_SIZE, 0);
            this.offscreenCtx.lineTo(x * this.CELL_SIZE, rows * this.CELL_SIZE);
            this.offscreenCtx.stroke();
        }
        
        for (let y = 0; y <= rows; y++) {
            this.offscreenCtx.beginPath();
            this.offscreenCtx.moveTo(0, y * this.CELL_SIZE);
            this.offscreenCtx.lineTo(cols * this.CELL_SIZE, y * this.CELL_SIZE);
            this.offscreenCtx.stroke();
        }
    }
}
