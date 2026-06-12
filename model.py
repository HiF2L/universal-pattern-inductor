# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

class UniversalMicroUnit(nn.Module):
    """
    A lightweight, multi-task neural micro-module that shares the core GRU and MLP layers
    across static and sequential tasks.
    
    Architecture:
    - Task-specific projections map different input shapes to the common hidden dimension.
    - The core shared weights consist of a tiny GRU layer followed by a Feedforward MLP.
    - Task-specific heads map the shared core representation to each task's output dimension.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # --- Task-Specific Input Projections ---
        # World 1: SpatialShiftWorld (25 flattened features to hidden_dim)
        self.proj_spatial_shift = nn.Linear(25, hidden_dim)
        
        # World 2: TemporalDelayWorld (1 feature per step to hidden_dim)
        self.proj_temporal_delay = nn.Linear(1, hidden_dim)
        
        # World 3: ContextInversionWorld (3 bits to hidden_dim)
        self.proj_context_inversion = nn.Linear(3, hidden_dim)
        
        # --- Shared Core Weights ---
        # The core recurrent layer (processes sequences of length 1 or 10)
        self.core_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # The core feedforward layer
        self.core_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Task-Specific Output Heads ---
        # World 1: SpatialShiftWorld (predicts center coordinate [x, y])
        self.head_spatial_shift = nn.Linear(hidden_dim, 2)
        
        # World 2: TemporalDelayWorld (predicts logits for sequence [T, 1])
        self.head_temporal_delay = nn.Linear(hidden_dim, 1)
        
        # World 3: ContextInversionWorld (predicts rule-based logic output [1])
        self.head_context_inversion = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
                - 'spatial_shift': Shape [B, 1, 5, 5]
                - 'temporal_delay': Shape [B, T, 1] (T = 10)
                - 'context_inversion': Shape [B, 3]
            task_name (str): One of 'spatial_shift', 'temporal_delay', or 'context_inversion'.

        Returns:
            torch.Tensor: Prediction tensor.
                - 'spatial_shift': Shape [B, 2]
                - 'temporal_delay': Shape [B, T, 1]
                - 'context_inversion': Shape [B, 1]
        """
        if task_name == 'spatial_shift':
            # --- Input Shape: [B, 1, 5, 5] ---
            # 1. Flatten the spatial dimensions to 25 features
            flat_x = x.view(x.size(0), -1)  # Shape: [B, 25]
            
            # 2. Project to shared hidden dimension
            proj_x = self.proj_spatial_shift(flat_x)  # Shape: [B, hidden_dim]
            
            # 3. Reshape to sequence of length 1
            seq_x = proj_x.unsqueeze(1)  # Shape: [B, 1, hidden_dim]
            
            # 4. Pass through shared GRU
            rnn_out, _ = self.core_rnn(seq_x)  # Shape: [B, 1, hidden_dim]
            
            # 5. Pass through shared MLP
            mlp_out = self.core_mlp(rnn_out)  # Shape: [B, 1, hidden_dim]
            
            # 6. Squeeze sequence dimension & pass through specific head
            features = mlp_out.squeeze(1)  # Shape: [B, hidden_dim]
            out = self.head_spatial_shift(features)  # Shape: [B, 2]
            return out
            
        elif task_name == 'temporal_delay':
            # --- Input Shape: [B, T, 1] where T = 10 ---
            # 1. Project step features to shared hidden dimension
            proj_x = self.proj_temporal_delay(x)  # Shape: [B, T, hidden_dim]
            
            # 2. Pass sequence through shared GRU
            rnn_out, _ = self.core_rnn(proj_x)  # Shape: [B, T, hidden_dim]
            
            # 3. Pass sequence through shared MLP
            mlp_out = self.core_mlp(rnn_out)  # Shape: [B, T, hidden_dim]
            
            # 4. Pass sequence through specific head
            out = self.head_temporal_delay(mlp_out)  # Shape: [B, T, 1]
            return out
            
        elif task_name == 'context_inversion':
            # --- Input Shape: [B, 3] ---
            # 1. Project bits to shared hidden dimension
            proj_x = self.proj_context_inversion(x)  # Shape: [B, hidden_dim]
            
            # 2. Reshape to sequence of length 1
            seq_x = proj_x.unsqueeze(1)  # Shape: [B, 1, hidden_dim]
            
            # 3. Pass through shared GRU
            rnn_out, _ = self.core_rnn(seq_x)  # Shape: [B, 1, hidden_dim]
            
            # 4. Pass through shared MLP
            mlp_out = self.core_mlp(rnn_out)  # Shape: [B, 1, hidden_dim]
            
            # 5. Squeeze sequence dimension & pass through specific head
            features = mlp_out.squeeze(1)  # Shape: [B, hidden_dim]
            out = self.head_context_inversion(features)  # Shape: [B, 1]
            return out
            
        else:
            raise ValueError(f"Unknown task name: {task_name}")


class IQMicroUnit(nn.Module):
    """
    Monolithic Universal Processor for the Procedural Fractal IQ-Engine (Transformer Version).
    Processes a sequence of 4 context pairs concatenated with the query vector, using
    Multi-Head Attention (nn.TransformerEncoder) to extract rules and predict the target query.
    
    Architecture:
    1. Universal Input Projector: maps context pairs of size 10 to hidden_dim=256.
    2. Query Projector: maps query input of size 5 to hidden_dim=256.
    3. Transformer Encoder Core: 3 layers of TransformerEncoder with 4 attention heads (batch_first=True).
    4. Universal Decoder Head: MLP mapping rule-infused query token representation of size 256 to size 5.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Universal Input Projector ---
        self.universal_projector = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # --- Query Projector ---
        self.query_projector = nn.Linear(5, hidden_dim)

        # --- Program Token Embedding ---
        self.program_embedding = nn.Embedding(26, hidden_dim)

        # --- Learnable Positional Embeddings ---
        # Positional embedding of shape [1, 10, hidden_dim] scaled by 0.02
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, hidden_dim) * 0.02)

        # --- Transformer Encoder Core ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.core_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- Universal Decoder Head ---
        # Maps the program slot representation to 26 class log-probabilities
        self.universal_decoder = nn.Linear(hidden_dim, 26)

    def forward(
        self, 
        X_context: torch.Tensor, 
        Y_context: torch.Tensor, 
        X_query: torch.Tensor,
        program_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for Transformer-based causal autoregressive program generation.

        Args:
            X_context (torch.Tensor): Context inputs. Shape: [B, T_ctx, 5]
            Y_context (torch.Tensor): Context targets. Shape: [B, T_ctx, 5]
            X_query (torch.Tensor): Query input. Shape: [B, 5]
            program_tokens (torch.Tensor, optional): Prior predicted program tokens. Shape: [B, S]

        Returns:
            torch.Tensor: Logits for the next tokens at program slots. Shape: [B, S, 26]
        """
        # 1. Project context pairs
        ctx_pairs = torch.cat([X_context, Y_context], dim=-1) # [B, T_ctx, 10]
        ctx_proj = self.universal_projector(ctx_pairs)       # [B, T_ctx, 256]
        T_ctx = ctx_proj.size(1)
        
        # 2. Project query
        qry_proj = self.query_projector(X_query).unsqueeze(1) # [B, 1, 256]
        
        # 3. Project program tokens (default to START=25 if None)
        B = X_context.size(0)
        device = X_context.device
        if program_tokens is None:
            program_tokens = torch.full((B, 1), 25, dtype=torch.long, device=device)
            
        prog_proj = self.program_embedding(program_tokens) # [B, S, 256]
        S = program_tokens.size(1)
        
        # 4. Concatenate along sequence dimension
        seq = torch.cat([ctx_proj, qry_proj, prog_proj], dim=1) # [B, T_ctx + 1 + S, 256]
        L = seq.size(1)
        
        # 5. Inject positional embeddings sequence-length agnostically
        pos_ctx = self.pos_embedding[:, :T_ctx, :]
        pos_qry = self.pos_embedding[:, T_ctx : T_ctx + 1, :]
        pos_prog = self.pos_embedding[:, T_ctx + 1 : T_ctx + 1 + S, :]
        pos = torch.cat([pos_ctx, pos_qry, pos_prog], dim=1)
        seq = seq + pos
        
        # 6. Construct dynamic causal mask of shape [L, L]
        mask = torch.full((L, L), float('-inf'), device=device)
        mask[:T_ctx+1, :T_ctx+1] = 0.0
        for i in range(T_ctx+1, L):
            mask[i, :i+1] = 0.0
            
        # 7. Transformer Encoder with Causal Mask
        out_seq = self.core_transformer(seq, mask=mask)
        
        # 8. Extract program slot outputs to classify the next token
        prog_out = out_seq[:, T_ctx + 1 : T_ctx + 1 + S, :] # [B, S, 256]
        logits = self.universal_decoder(prog_out)   # [B, S, 26]
        
        return logits
