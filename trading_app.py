import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Configuration forcée pour CPU
device = torch.device("cpu")
torch.set_default_device("cpu")

# Configuration du trading
INITIAL_INVESTMENT = 10_000
CFD_CONTRACT_SIZE = 25_000
LEVERAGE = 20
SPREAD = 0.002
SWAP_FEE_PER_NIGHT = 0.005
IBKR_CFD_COMMISSION = 2.00
MARKET_DATA_FEE = 10
RISK_PER_TRADE = 0.02 * INITIAL_INVESTMENT
MARGIN_REQUIRED_PER_CONTRACT = (CFD_CONTRACT_SIZE * 4.00) / LEVERAGE
MAX_DAILY_TRADES = 5

# Chargement des données
df_copper_rl = pd.read_csv('df_copper_rl.csv', index_col='date', parse_dates=True)
df_encoded = pd.read_csv('df_encoded.csv', index_col='date', parse_dates=True)

# Définition du modèle Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

# Fonctions utilitaires
def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_eps=0.2, entropy_coef=0.01):
    for _ in range(4):  
        probs, values = model(states)
        dist = Categorical(probs)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - log_probs_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - values.squeeze()).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

def get_action(model, state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        probs, value = model(state)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), probs.detach().cpu().numpy().flatten(), value.item()

def compute_reward(pnl, prev_pnl, position_change=False):
    base_reward = pnl - prev_pnl
    if position_change:
        base_reward -= 1.0
    return base_reward

def get_market_price(true_market_data):
    return true_market_data.iloc[0] if hasattr(true_market_data, 'iloc') else true_market_data[0]

def create_features_from_historical(historical_data, n_features):
    if historical_data.shape[1] >= n_features:
        return historical_data.iloc[-1, :n_features].values
    else:
        base_features = historical_data.iloc[-1].values
        additional_features = []
        
        if len(historical_data) >= 5:
            additional_features.append(historical_data.iloc[-5:, 0].mean())  # MA5
        if len(historical_data) >= 10:
            additional_features.append(historical_data.iloc[-10:, 0].mean())  # MA10
        if len(historical_data) >= 20:
            additional_features.append(historical_data.iloc[-20:, 0].mean())  # MA20
            
        if len(historical_data) >= 5:
            additional_features.append(historical_data.iloc[-5:, 0].std())
            
        while len(base_features) + len(additional_features) < n_features:
            additional_features.append(0.0)
            
        combined_features = np.concatenate([base_features, additional_features])
        return combined_features[:n_features]

# Configuration de l'application Streamlit
st.set_page_config(page_title="Trading PPO Dashboard", layout="wide")

st.title("Trading PPO Dashboard")
st.markdown("**Apprentissage par renforcement pour le trading de CFD sur le cuivre**")

# Vérification des données
if 'df_copper_rl' not in st.session_state or 'df_encoded' not in st.session_state:
    st.error("Les DataFrames df_copper_rl et df_encoded doivent être chargés dans votre notebook avant d'exécuter cette application.")
    st.info("Assurez-vous que les variables df_copper_rl et df_encoded sont disponibles dans votre environnement.")
    st.stop()

# Chargement des données depuis les variables globales
@st.cache_data
def load_data():
    try:
        # Accès aux variables globales du notebook
        import __main__
        df_copper_rl = __main__.df_copper_rl
        df_encoded = __main__.df_encoded
        return df_copper_rl, df_encoded
    except:
        st.error("Impossible de charger les données. Assurez-vous que df_copper_rl et df_encoded sont définis.")
        return None, None

df_copper_rl, df_encoded = load_data()

if df_copper_rl is None or df_encoded is None:
    st.stop()

# Sidebar pour les paramètres
st.sidebar.header("⚙️ Paramètres")

# Paramètres du modèle
st.sidebar.subheader("Modèle PPO")
n_epochs = st.sidebar.slider("Nombre d'époques d'entraînement", 5, 50, 15)
hidden_dim = st.sidebar.slider("Dimension cachée", 64, 256, 128)
learning_rate = st.sidebar.select_slider("Taux d'apprentissage", 
                                         options=[1e-4, 3e-4, 1e-3, 3e-3], 
                                         value=3e-4, 
                                         format_func=lambda x: f"{x:.0e}")

# Paramètres de trading
st.sidebar.subheader("Trading")
initial_investment = st.sidebar.number_input("Investissement initial ($)", 
                                            min_value=1000, 
                                            max_value=100000, 
                                            value=INITIAL_INVESTMENT)

# Préparation des données
historical_df = df_copper_rl[df_copper_rl.index <= "2025-04-11"].copy()
df_encoded_filtered = df_encoded[(df_encoded.index >= "2025-04-14") & (df_encoded.index <= "2025-04-25")].copy()
df_future_true = df_copper_rl[(df_copper_rl.index >= "2025-04-14") & (df_copper_rl.index <= "2025-04-25")].copy()

# Affichage des informations sur les données
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Données historiques", f"{len(historical_df)} jours")
with col2:
    st.metric("Données de test", f"{len(df_encoded_filtered)} jours")
with col3:
    st.metric("Dimensions d'entrée", df_encoded_filtered.shape[1])

# Bouton pour lancer l'entraînement
if st.button("Lancer l'entraînement et le trading", type="primary"):
    # Initialisation du modèle
    input_dim = df_encoded_filtered.shape[1]
    ppo_model = ActorCritic(input_dim, hidden_dim).to(device)
    optimizer = optim.Adam(ppo_model.parameters(), lr=learning_rate)
    
    # Barre de progression pour l'entraînement
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Entraînement
    train_data = historical_df.copy()
    gamma = 0.99
    lam = 0.95
    
    status_text.text("Entraînement en cours...")
    
    for epoch in range(n_epochs):
        states, actions, rewards, log_probs, values, masks = [], [], [], [], [], []
        
        position = 0
        entry_price = 0
        portfolio_value = initial_investment
        
        for i in range(len(train_data)-1):
            state_features = create_features_from_historical(train_data.iloc[:i+1], input_dim)
            state = torch.FloatTensor(state_features).to(device)
            
            next_price = train_data.iloc[i+1, 0]
            
            probs, value = ppo_model(state)
            dist = Categorical(probs)
            action = dist.sample()
            
            reward = 0
            
            if action.item() == 1 and position == 0:  # Buy
                position = 1
                entry_price = next_price
                reward = -IBKR_CFD_COMMISSION
            elif action.item() == 2 and position == 0:  # Sell
                position = -1
                entry_price = next_price
                reward = -IBKR_CFD_COMMISSION
            elif action.item() == 0 and position != 0:  # Close position
                pnl = (next_price - entry_price) * CFD_CONTRACT_SIZE * position
                reward = pnl - IBKR_CFD_COMMISSION
                position = 0
            
            reward = reward / 1000.0
            done = 0 if i < len(train_data) - 2 else 1
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            values.append(value.item())
            masks.append(1 - done)
        
        if len(states) > 0:
            returns = compute_gae(rewards, values, masks, gamma, lam)
            advantages = torch.FloatTensor(returns).to(device) - torch.FloatTensor(values).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ppo_update(
                model=ppo_model,
                optimizer=optimizer,
                states=torch.stack(states),
                actions=torch.tensor(actions).to(device),
                log_probs_old=torch.stack(log_probs).detach(),
                returns=torch.tensor(returns).to(device),
                advantages=advantages.detach()
            )
        
        progress_bar.progress((epoch + 1) / n_epochs)
        
        if (epoch + 1) % 5 == 0:
            avg_reward = np.mean(rewards) if rewards else 0
            status_text.text(f"Époque {epoch+1}/{n_epochs} - Récompense moyenne: {avg_reward:.4f}")
    
    status_text.text("Entraînement terminé! Début du trading...")
    
    # Trading incrémental
    updated_historical_df = historical_df.copy()
    portfolio_value = initial_investment
    position = 0
    entry_price = 0
    results = []
    
    online_buffer = {
        'states': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'masks': []
    }
    
    for i, date in enumerate(df_encoded_filtered.index):
        encoded_features = df_encoded_filtered.loc[date].values
        action, probs, value_est = get_action(ppo_model, encoded_features)
        action_str = ['hold', 'buy', 'sell'][action]
        
        true_market_data = df_future_true.loc[date]
        market_price = get_market_price(true_market_data)
        
        prev_portfolio = portfolio_value
        pnl = 0
        
        if action_str == 'buy' and position == 0:
            position = 1
            entry_price = market_price
            portfolio_value -= IBKR_CFD_COMMISSION
        elif action_str == 'sell' and position == 0:
            position = -1
            entry_price = market_price
            portfolio_value -= IBKR_CFD_COMMISSION
        elif action_str == 'hold' and position != 0:
            pnl = (market_price - entry_price) * CFD_CONTRACT_SIZE * position
            portfolio_value += pnl - IBKR_CFD_COMMISSION
            position = 0
        
        reward = compute_reward(portfolio_value, prev_portfolio)
        
        results.append({
            "date": date,
            "market_price": market_price,
            "action": action_str,
            "action_prob_hold": probs[0],
            "action_prob_buy": probs[1],
            "action_prob_sell": probs[2],
            "position": position,
            "entry_price": entry_price if position != 0 else None,
            "pnl": pnl,
            "portfolio_value": portfolio_value,
            "reward": reward,
            "value_estimation": value_est
        })
        
        updated_historical_df.loc[date] = true_market_data
        
        # Réentraînement en ligne
        state_tensor = torch.FloatTensor(encoded_features).to(device)
        action_tensor = torch.tensor(action).to(device)
        
        probs_tensor, value_tensor = ppo_model(state_tensor.unsqueeze(0))
        dist = Categorical(probs_tensor)
        log_prob = dist.log_prob(action_tensor)
        
        online_buffer['states'].append(state_tensor)
        online_buffer['actions'].append(action_tensor)
        online_buffer['rewards'].append(reward)
        online_buffer['log_probs'].append(log_prob)
        online_buffer['values'].append(value_tensor.item())
        online_buffer['masks'].append(1.0)
        
        if len(online_buffer['states']) >= 3:
            states_batch = torch.stack(online_buffer['states'])
            actions_batch = torch.tensor(online_buffer['actions']).to(device)
            rewards_batch = online_buffer['rewards']
            log_probs_batch = torch.stack(online_buffer['log_probs'])
            values_batch = online_buffer['values']
            masks_batch = online_buffer['masks']
            
            returns = compute_gae(rewards_batch, values_batch, masks_batch, gamma, lam)
            advantages = torch.FloatTensor(returns).to(device) - torch.FloatTensor(values_batch).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ppo_update(
                model=ppo_model,
                optimizer=optimizer,
                states=states_batch,
                actions=actions_batch,
                log_probs_old=log_probs_batch.detach(),
                returns=torch.tensor(returns).to(device),
                advantages=advantages.detach()
            )
            
            online_buffer = {k: [] for k in online_buffer.keys()}
    
    # Fermeture de position finale
    if position != 0:
        final_price = market_price
        final_pnl = (final_price - entry_price) * CFD_CONTRACT_SIZE * position
        portfolio_value += final_pnl - IBKR_CFD_COMMISSION
        results[-1]["portfolio_value"] = portfolio_value
        results[-1]["pnl"] = final_pnl
    
    # Création du DataFrame des résultats
    df_results = pd.DataFrame(results).set_index("date")
    
    # Sauvegarde des résultats dans session_state
    st.session_state['df_results'] = df_results
    st.session_state['portfolio_value'] = portfolio_value
    st.session_state['ppo_model'] = ppo_model
    
    status_text.text("Trading terminé!")
    progress_bar.progress(1.0)

# Affichage des résultats
if 'df_results' in st.session_state:
    df_results = st.session_state['df_results']
    final_portfolio_value = st.session_state['portfolio_value']
    
    st.success("Analyse terminée avec succès!")
    
    # Métriques principales
    st.subheader("Résultats de Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Valeur finale", 
            f"${final_portfolio_value:,.2f}",
            f"${final_portfolio_value - initial_investment:,.2f}"
        )
    
    with col2:
        total_return = ((final_portfolio_value - initial_investment) / initial_investment) * 100
        st.metric("Rendement total", f"{total_return:.2f}%")
    
    with col3:
        total_trades = len(df_results[df_results['action'] != 'hold'])
        st.metric("Nombre de trades", total_trades)
    
    with col4:
        avg_pnl = df_results['pnl'].mean()
        st.metric("P&L moyen", f"${avg_pnl:.2f}")
    
    # Graphiques
    st.subheader("Visualisations")
    
    # Évolution du portefeuille
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_results.index, 
        y=df_results['portfolio_value'], 
        mode='lines+markers',
        name='Valeur du Portefeuille',
        line=dict(color='blue', width=2)
    ))
    fig1.add_hline(y=initial_investment, line_dash="dash", line_color="red", 
                  annotation_text="Investissement Initial")
    fig1.update_layout(
        title="Évolution de la Valeur du Portefeuille",
        xaxis_title="Date",
        yaxis_title="Valeur ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Actions vs Prix du marché
    fig2 = go.Figure()
    colors = {'buy': 'green', 'sell': 'red', 'hold': 'gray'}
    
    for action in ['buy', 'sell', 'hold']:
        action_mask = df_results['action'] == action
        if action_mask.any():
            fig2.add_trace(go.Scatter(
                x=df_results.index[action_mask],
                y=df_results['market_price'][action_mask],
                mode='markers',
                name=action.capitalize(),
                marker=dict(color=colors[action], size=10),
                text=df_results['action'][action_mask],
                hovertemplate='%{text}<br>Prix: %{y:.4f}<extra></extra>'
            ))
    
    fig2.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['market_price'],
        mode='lines',
        name='Prix du Marché',
        line=dict(color='black', width=1)
    ))
    
    fig2.update_layout(
        title="Actions de Trading vs Prix du Marché",
        xaxis_title="Date",
        yaxis_title="Prix du Marché",
        showlegend=True
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Probabilités d'action
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['action_prob_hold'],
        mode='lines',
        name='Probabilité Hold',
        line=dict(color='gray')
    ))
    fig3.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['action_prob_buy'],
        mode='lines',
        name='Probabilité Buy',
        line=dict(color='green')
    ))
    fig3.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['action_prob_sell'],
        mode='lines',
        name='Probabilité Sell',
        line=dict(color='red')
    ))
    
    fig3.update_layout(
        title="Probabilités des Actions par le Modèle PPO",
        xaxis_title="Date",
        yaxis_title="Probabilité",
        showlegend=True
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Tableau des résultats détaillés
    st.subheader("Résultats Détaillés")
    st.dataframe(df_results, use_container_width=True)
    
    # Téléchargement des résultats
    csv = df_results.to_csv()
    st.download_button(
        label="💾 Télécharger les résultats (CSV)",
        data=csv,
        file_name=f'trading_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Cliquez sur le bouton ci-dessus pour lancer l'entraînement et le trading.")
    
    # Aperçu des données
    st.subheader("Aperçu des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Données historiques (échantillon):**")
        st.dataframe(historical_df.head(), use_container_width=True)
    
    with col2:
        st.write("**Données encodées (échantillon):**")
        st.dataframe(df_encoded_filtered.head(), use_container_width=True)
    
    # Graphique des prix historiques
    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df.iloc[:, 0],
        mode='lines',
        name='Prix Historique',
        line=dict(color='blue')
    ))
    fig_preview.update_layout(
        title="Prix Historique du Cuivre",
        xaxis_title="Date",
        yaxis_title="Prix"
    )
    st.plotly_chart(fig_preview, use_container_width=True)