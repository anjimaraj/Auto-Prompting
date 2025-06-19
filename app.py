def main():
    st.set_page_config(
        page_title="Advanced PPO Text Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv("GROQ_API_KEY", "")
    
    st.title("🧠 Advanced PPO Text Classification System")
    st.markdown("*Reinforcement Learning-Enhanced Text Classification with LLM Integration*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙ Configuration")
        
        # API Key
        api_key = st.text_input(
            "Groq API Key", 
            value=st.session_state.api_key, 
            type="password",
            help="Your Groq API key for LLM integration"
        )
        st.session_state.api_key = api_key
        
        # Test API connection
        if api_key:
            if st.button("🔗 Test Connection"):
                with st.spinner("Testing..."):
                    try:
                        client = GroqAPIClient(api_key)
                        response = client.get_completion("Test")
                        st.success("✅ Connection successful!")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
        
        st.divider()
        
        # Training parameters
        st.subheader("🎯 Training Configuration")
        num_episodes = st.slider("Training Episodes", 3, 50, 10)
        batch_size = st.slider("Batch Size", 2, 10, 4)
        sample_size = st.slider("Evaluation Sample Size", 10, 100, 30)
        
        st.divider()
        
        # System actions
        st.subheader("💾 System Management")
        if st.button("🔄 Reset System"):
            st.session_state.classifier = None
            st.session_state.training_history = []
            st.success("System reset!")
            st.rerun()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Management", 
        "🎓 Training Center", 
        "🔮 Prediction Engine",
        "📈 Analytics Dashboard",
        "ℹ System Info"
    ])
    
    with tab1:
        data_management_tab()
    
    with tab2:
        training_center_tab(api_key, num_episodes, batch_size)
    
    with tab3:
        prediction_engine_tab()
    
    with tab4:
        analytics_dashboard_tab(sample_size)
    
    with tab5:
        system_info_tab()

def data_management_tab():
    st.header("📊 Data Management Center")
    
    col1, col2 = st.columns([3, 1])
    
    # Sample data loading
    with col2:
        if st.button("📋 Load Sample Data", type="primary"):
            sample_data = [
                {'text': 'Revolutionary renewable energy breakthrough promises to transform our climate future with unprecedented efficiency gains.', 'sentiment': 'positive', 'category': 'environment'},
                {'text': 'Global financial markets plummeted today as investors fled risky assets amid mounting economic uncertainties.', 'sentiment': 'negative', 'category': 'financial'},
                {'text': 'Government officials announced comprehensive policy reforms to address growing concerns about democratic institutions.', 'sentiment': 'neutral', 'category': 'governance'},
                {'text': 'Latest smartphone app update introduces innovative features but unfortunately breaks compatibility with older devices.', 'sentiment': 'negative', 'category': 'noise'},
                {'text': 'Community leaders celebrate successful initiative bringing diverse neighborhoods together through cultural exchange programs.', 'sentiment': 'positive', 'category': 'social'},
                {'text': 'Diplomatic tensions escalate as international negotiations reach critical juncture over territorial disputes.', 'sentiment': 'negative', 'category': 'geopolitical'},
                {'text': 'New environmental protection measures receive mixed reactions from industry stakeholders and conservation groups.', 'sentiment': 'neutral', 'category': 'environment'},
                {'text': 'Economic indicators suggest cautious optimism as quarterly earnings exceed analyst expectations across sectors.', 'sentiment': 'positive', 'category': 'financial'},
                {'text': 'Social media platform implements enhanced privacy controls following user feedback and regulatory pressure.', 'sentiment': 'neutral', 'category': 'noise'},
                {'text': 'International cooperation strengthens as nations unite to address shared challenges through multilateral agreements.', 'sentiment': 'positive', 'category': 'geopolitical'}
            ]
            st.session_state.training_data = sample_data
            st.success(f"✅ Loaded {len(sample_data)} high-quality training samples!")
            st.rerun()
    
    # File upload
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Training Data (CSV Format)",
            type=['csv'],
            help="CSV with columns: text, sentiment, category"
        )
    
    # File processing
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            if len(lines) < 2:
                st.error("❌ File needs header and data rows")
                return
            
            # Parse CSV
            header = [col.strip().strip('"') for col in lines[0].split(',')]
            data = []
            
            for line in lines[1:]:
                if line.strip():
                    values = [val.strip().strip('"') for val in line.split(',')]
                    if len(values) >= len(header):
                        data.append(dict(zip(header, values)))
            
            st.success(f"✅ Processed {len(data)} rows from uploaded file")
            
            # Validate data
            valid_data = []
            for item in data:
                text = str(item.get('text', '')).strip()
                sentiment = str(item.get('sentiment', '')).strip()
                category = str(item.get('category', '')).strip()
                
                if len(text) > 10 and sentiment and category:
                    valid_data.append({
                        'text': text,
                        'sentiment': sentiment.lower(),
                        'category': category.lower()
                    })
            
            if valid_data:
                st.session_state.training_data = valid_data
                st.success(f"✅ {len(valid_data)} valid samples ready for training")
            else:
                st.error("❌ No valid data found")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    # Display current data
    if st.session_state.training_data:
        st.subheader("📋 Current Dataset")
        
        data = st.session_state.training_data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            sentiments = [item['sentiment'] for item in data]
            unique_sentiments = len(set(sentiments))
            st.metric("Sentiment Classes", unique_sentiments)
        with col3:
            categories = [item['category'] for item in data]
            unique_categories = len(set(categories))
            st.metric("Category Classes", unique_categories)
        
        # Distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = {}
            for item in data:
                sentiment = item['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(data)) * 100
                st.write(f"• *{sentiment.title()}*: {count} ({percentage:.1f}%)")
        
        with col2:
            st.subheader("Category Distribution")
            category_counts = {}
            for item in data:
                category = item['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            for category, count in category_counts.items():
                percentage = (count / len(data)) * 100
                st.write(f"• *{category.title()}*: {count} ({percentage:.1f}%)")
        
        # Sample preview
        st.subheader("📖 Sample Preview")
        preview_count = min(3, len(data))
        for i in range(preview_count):
            item = data[i]
            with st.expander(f"Sample {i+1}: {item['sentiment']} | {item['category']}"):
                st.write(f"*Text*: {item['text']}")

def training_center_tab(api_key, num_episodes, batch_size):
    st.header("🎓 Training Center")
    
    if not api_key:
        st.warning("⚠ Groq API key required for training")
        return
    
    if not st.session_state.training_data:
        st.warning("⚠ Upload training data first")
        return
    
    # Training controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🚀 Training Controls")
        
        if st.session_state.classifier is None:
            if st.button("🔧 Initialize Advanced Classifier", type="primary"):
                with st.spinner("Initializing system..."):
                    try:
                        st.session_state.classifier = AdvancedTextClassifier(
                            api_key=api_key,
                            few_shot_data=st.session_state.training_data[:8]
                        )
                        st.success("✅ Advanced classifier initialized!")
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {e}")
                        return
    
    with col2:
        st.metric("📊 Data Samples", len(st.session_state.training_data))
    
    with col3:
        st.metric("🎯 Episodes", num_episodes)
    
    # Training execution
    if st.session_state.classifier is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎓 Start Training", type="primary"):
                run_training_session(num_episodes, batch_size)
        
        with col2:
            if st.button("🔄 Quick Training (3 episodes)"):
                run_training_session(3, min(batch_size, 3))
    
    # Training history visualization
    if st.session_state.training_history:
        st.subheader("📈 Training Progress")
        
        history = st.session_state.training_history
        episodes = [h['episode'] for h in history]
        rewards = [h['reward'] for h in history]
        accuracies = [h['accuracy'] for h in history]
        
        # Progress charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reward Progress")
            chart_data = {}
            for ep, reward in zip(episodes, rewards):
                chart_data[f"Ep {ep}"] = reward
            st.bar_chart(chart_data)
        
        with col2:
            st.subheader("Accuracy Progress")
            chart_data = {}
            for ep, acc in zip(episodes, accuracies):
                chart_data[f"Ep {ep}"] = acc
            st.bar_chart(chart_data)
        
        # Latest metrics
        latest = history[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Latest Reward", f"{latest['reward']:.3f}")
        with col2:
            st.metric("Latest Accuracy", f"{latest['accuracy']:.3f}")
        with col3:
            st.metric("Total Episodes", len(history))
        with col4:
            avg_reward = sum(rewards) / len(rewards)
            st.metric("Avg Reward", f"{avg_reward:.3f}")

def run_training_session(num_episodes, batch_size):
    """Execute comprehensive training session"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    training_data = st.session_state.training_data
    classifier = st.session_state.classifier
    
    for episode in range(num_episodes):
        start_time = time.time()
        
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        status_text.text(f"Training Episode {episode + 1}/{num_episodes}")
        
        try:
            # Sample training batch
            if len(training_data) >= batch_size:
                batch = random.sample(training_data, batch_size)
            else:
                batch = training_data
            
            # Execute training episode
            avg_reward, avg_accuracy = classifier.train_episode(batch)
            duration = time.time() - start_time
            
            # Store training history
            episode_data = {
                'episode': episode + 1,
                'reward': avg_reward,
                'accuracy': avg_accuracy,
                'duration': duration,
                'timestamp': time.time(),
                'batch_size': len(batch)
            }
            st.session_state.training_history.append(episode_data)
            
            # Real-time metrics display
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Reward", f"{avg_reward:.3f}")
                with col2:
                    st.metric("Current Accuracy", f"{avg_accuracy:.3f}")
                with col3:
                    st.metric("Duration", f"{duration:.1f}s")
                with col4:
                    st.metric("Batch Size", len(batch))
            
            # Rate limiting for API
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"❌ Episode {episode + 1} failed: {e}")
            break
    
    progress_bar.progress(1.0)
    status_text.text("✅ Training session completed!")
    st.success(f"🎉 Completed {num_episodes} episodes successfully!")

def prediction_engine_tab():
    st.header("🔮 Prediction Engine")
    
    if st.session_state.classifier is None:
        st.warning("⚠ Initialize and train classifier first")
        return
    
    # Prediction interface
    st.subheader("📝 Text Classification")
    
    # Input methods
    input_method = st.radio(
        "Input Method:",
        ["Manual Text Entry", "Example Selector"],
        horizontal=True
    )
    
    if input_method == "Manual Text Entry":
        text_input = st.text_area(
            "Enter text to classify:",
            height=120,
            placeholder="Type or paste your text here for sentiment and category analysis..."
        )
        
        if st.button("🔮 Analyze Text", type="primary") and text_input.strip():
            analyze_text(text_input.strip())
    
    else:
        # Example selector
        examples = [
            "Revolutionary breakthrough in renewable energy storage promises to accelerate global transition to clean power.",
            "Major financial institutions report significant losses due to unexpected market volatility and regulatory changes.",
            "Government announces comprehensive immigration reform package after months of bipartisan negotiations.",
            "New social media platform gains millions of users but faces criticism over data privacy policies.",
            "International summit concludes with historic agreement on climate action and sustainable development goals.",
            "Cutting-edge artificial intelligence system demonstrates remarkable capabilities in medical diagnosis applications."
        ]
        
        selected_example = st.selectbox("Choose an example:", examples)
        
        if st.button("🔮 Analyze Example", type="primary"):
            analyze_text(selected_example)
    
    # Batch prediction
    st.subheader("📦 Batch Processing")
    
    batch_texts = st.text_area(
        "Enter multiple texts (one per line):",
        height=100,
        placeholder="Line 1: First text to analyze\nLine 2: Second text to analyze\n..."
    )
    
    if st.button("🔮 Analyze Batch") and batch_texts.strip():
        texts = [line.strip() for line in batch_texts.strip().split('\n') if line.strip()]
        if texts:
            analyze_batch(texts)

def analyze_text(text):
    """Analyze single text with comprehensive results"""
    with st.spinner("Analyzing text..."):
        try:
            prediction = st.session_state.classifier.predict(text)
            
            # Main results
            st.subheader("📊 Classification Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_colors = {
                    'positive': '🟢',
                    'negative': '🔴', 
                    'neutral': '🟡'
                }
                icon = sentiment_colors.get(prediction['sentiment'], '⚪')
                st.metric(f"{icon} Sentiment", prediction['sentiment'].title())
            
            with col2:
                category_icons = {
                    'environment': '🌍',
                    'financial': '💰',
                    'geopolitical': '🌐',
                    'governance': '🏛',
                    'social': '👥',
                    'noise': '📱'
                }
                icon = category_icons.get(prediction['category'], '📂')
                st.metric(f"{icon} Category", prediction['category'].title())
            
            with col3:
                confidence = prediction['confidence']
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                st.metric(f"{color} Confidence", f"{confidence:.1%}")
            
            # Detailed analysis
            st.subheader("🔍 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("*Strategy Used:*", prediction.get('strategy_used', 'Unknown'))
                st.write("*Cache Hit:*", "✅ Yes" if prediction.get('cache_hit', False) else "❌ No")
            
            with col2:
                confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
                st.write("*Confidence Level:*", confidence_level)
                st.write("*Processing:*", "Real-time API" if not prediction.get('cache_hit') else "Cached result")
            
            # Raw prediction data
            with st.expander("🔧 Technical Details"):
                st.json(prediction)
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")

def analyze_batch(texts):
    """Analyze multiple texts with summary results"""
    with st.spinner(f"Analyzing {len(texts)} texts..."):
        try:
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                prediction = st.session_state.classifier.predict(text)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': prediction['sentiment'],
                    'category': prediction['category'],
                    'confidence': prediction['confidence']
                })
                progress_bar.progress((i + 1) / len(texts))
            
            progress_bar.empty()
            
            # Results summary
            st.subheader("📊 Batch Results Summary")
            
            # Aggregate statistics
            sentiments = [r['sentiment'] for r in results]
            categories = [r['category'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_dist = {s: sentiments.count(s) for s in set(sentiments)}
                st.write("*Sentiment Distribution:*")
                for sentiment, count in sentiment_dist.items():
                    st.write(f"• {sentiment.title()}: {count}")
            
            with col2:
                category_dist = {c: categories.count(c) for c in set(categories)}
                st.write("*Category Distribution:*")
                for category, count in category_dist.items():
                    st.write(f"• {category.title()}: {count}")
            
            with col3:
                avg_confidence = sum(confidences) / len(confidences)
                high_conf = sum(1 for c in confidences if c > 0.8)
                st.write("*Confidence Analysis:*")
                st.write(f"• Average: {avg_confidence:.1%}")
                st.write(f"• High confidence: {high_conf}/{len(confidences)}")
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}: {result['sentiment']} | {result['category']}"):
                    st.write(f"*Text:* {result['text']}")
                    st.write(f"*Sentiment:* {result['sentiment'].title()}")
                    st.write(f"*Category:* {result['category'].title()}")
                    st.write(f"*Confidence:* {result['confidence']:.1%}")
                    
        except Exception as e:
            st.error(f"❌ Batch analysis failed: {e}")

def analytics_dashboard_tab(sample_size):
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.classifier is None:
        st.warning("⚠ No classifier available for analysis")
        return
    
    # System performance metrics
    st.subheader("🔧 System Performance")
    
    classifier = st.session_state.classifier
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cache_total = classifier.cache_hits + classifier.cache_misses
        cache_rate = classifier.cache_hits / cache_total if cache_total > 0 else 0
        st.metric("Cache Hit Rate", f"{cache_rate:.1%}")
    
    with col2:
        st.metric("Training Steps", classifier.total_training_steps)
    
    with col3:
        st.metric("Cached Predictions", len(classifier.prediction_cache))
    
    with col4:
        strategy_count = len(classifier.strategy_performance)
        st.metric("Active Strategies", strategy_count)
    
    # Strategy performance analysis
    if classifier.strategy_performance:
        st.subheader("🎯 Strategy Performance Analysis")
        
        strategy_data = []
        for strategy, rewards in classifier.strategy_performance.items():
            if rewards:
                strategy_data.append({
                    'strategy': strategy,
                    'avg_reward': sum(rewards) / len(rewards),
                    'usage_count': classifier.strategy_usage[strategy],
                    'total_reward': sum(rewards),
                    'best_reward': max(rewards),
                    'worst_reward': min(rewards)
                })
        
        if strategy_data:
            # Sort by average reward
            strategy_data.sort(key=lambda x: x['avg_reward'], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("*Top Strategies by Average Reward:*")
                for i, data in enumerate(strategy_data[:5], 1):
                    st.write(f"{i}. *{data['strategy']}*: {data['avg_reward']:.3f} ({data['usage_count']} uses)")
            
            with col2:
                st.write("*Strategy Usage Distribution:*")
                total_usage = sum(data['usage_count'] for data in strategy_data)
                for data in strategy_data[:5]:
                    percentage = (data['usage_count'] / total_usage) * 100
                    st.write(f"• *{data['strategy']}*: {percentage:.1f}%")
    
    # Model evaluation
    if st.session_state.training_data:
        st.subheader("🧪 Model Evaluation")
        
        if st.button("🔬 Run Comprehensive Evaluation"):
            with st.spinner("Evaluating model performance..."):
                try:
                    eval_results = classifier.evaluate(st.session_state.training_data, sample_size)
                    
                    if eval_results:
                        # Performance metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment Accuracy", f"{eval_results['sentiment_accuracy']:.1%}")
                        with col2:
                            st.metric("Category Accuracy", f"{eval_results['category_accuracy']:.1%}")
                        with col3:
                            st.metric("Overall Accuracy", f"{eval_results['both_correct']:.1%}")
                        
                        # Additional metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg Confidence", f"{eval_results['avg_confidence']:.1%}")
                        with col2:
                            st.metric("Avg Processing Time", f"{eval_results['avg_processing_time']:.2f}s")
                        with col3:
                            st.metric("Samples Evaluated", eval_results['total_samples'])
                        
                        # Strategy distribution in evaluation
                        if 'strategy_distribution' in eval_results:
                            st.subheader("📊 Strategy Usage in Evaluation")
                            strategy_dist = eval_results['strategy_distribution']
                            for strategy, count in sorted(strategy_dist.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / eval_results['total_samples']) * 100
                                st.write(f"• *{strategy}*: {count} uses ({percentage:.1f}%)")
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
    
    # Training history analysis
    if st.session_state.training_history:
        st.subheader("📈 Training History Analysis")
        
        history = st.session_state.training_history
        
        # Performance trends
        col1, col2 = st.columns(2)
        
        with col1:
            rewards = [h['reward'] for h in history]
            st.write("*Reward Statistics:*")
            st.write(f"• Best Episode: {max(rewards):.3f}")
            st.write(f"• Average: {sum(rewards)/len(rewards):.3f}")
            st.write(f"• Latest: {rewards[-1]:.3f}")
            
            # Improvement trend
            if len(rewards) >= 5:
                recent_avg = sum(rewards[-5:]) / 5
                early_avg = sum(rewards[:5]) / 5
                improvement = ((recent_avg - early_avg) / early_avg) * 100
                st.write(f"• Improvement: {improvement:+.1f}%")
        
        with col2:
            accuracies = [h['accuracy'] for h in history]
            st.write("*Accuracy Statistics:*")
            st.write(f"• Best Episode: {max(accuracies):.1%}")
            st.write(f"• Average: {sum(accuracies)/len(accuracies):.1%}")
            st.write(f"• Latest: {accuracies[-1]:.1%}")
            
            # Consistency measure
            perfect_episodes = sum(1 for acc in accuracies if acc == 1.0)
            st.write(f"• Perfect Episodes: {perfect_episodes}/{len(accuracies)}")

import streamlit as st

def system_info_tab():
    st.header("ℹ️ System Information")
    
    st.markdown("""
    ## 🧠 Advanced PPO Text Classification System
    
    This system implements a sophisticated approach to text classification using:

    ### 🔧 Core Technologies

    **Reinforcement Learning Framework:**
    - 🧮 PPO (Proximal Policy Optimization) for strategy learning
    - 🎭 Actor-Critic Architecture with neural simulation
    - ⚖️ Generalized Advantage Estimation for stable training
    - ♻️ Experience Replay and intelligent caching

    **Text Processing Pipeline:**
    - 🧩 Advanced feature engineering with 25+ linguistic features
    - 🧠 Multi-strategy prompting with 12 distinct styles
    - 🔍 Context-aware prompt optimization
    - 🧠 Smart caching for performance

    **LLM Integration:**
    - 🚀 Groq API using LLaMA3-8b-8192 model
    - 🧾 Structured JSON output for consistency
    - 🔁 Retry logic and exception handling
    - 🧭 Rate limiting compliance

    ### 📊 Classification Capabilities

    **Sentiment Analysis**
    - 🟢 Positive: Favorable, optimistic, or supportive tone
    - 🔴 Negative: Critical, pessimistic, or adversarial tone
    - ⚪ Neutral: Balanced, factual, or objective tone

    **Category Classification**
    - 🌍 Environment: Sustainability, climate, ecology
    - 💰 Financial: Economy, stocks, business
    - 🏛 Governance: Politics, regulations, government
    - 👥 Social: Society, culture, relationships
    - 🌐 Geopolitical: International affairs
    - 📱 Noise: Miscellaneous, tech, irrelevant topics

    ### 🚀 Key Features

    **Learning & Adaptation**
    - Self-optimizing strategy selection via RL
    - Real-time feedback for continuous learning
    - Strategy performance tracking

    **Performance & Efficiency**
    - Caching reduces redundant API calls
    - Efficient batch processing
    - Real-time analytics support

    **User Experience**
    - Streamlit web interface
    - Immediate testing via sample or uploaded data
    - Comprehensive reporting

    ### 🔬 Technical Architecture

    **Training Process**
    1. Encode text into feature vectors
    2. PPO agent selects optimal strategy
    3. LLM classifies using generated prompt
    4. Accuracy & confidence-based reward
    5. PPO policy update

    **Prediction Process**
    1. Feature extraction
    2. Strategy selection via agent
    3. Prompt generation
    4. LLM completion via API
    5. Structured output parsing

    ### 📝 Data Format Requirements

    Upload your CSV file with **at least the following columns**:

    - `text`: The input text to be classified
    - `sentiment`: Ground truth sentiment (`positive`, `negative`, `neutral`)
    - `category`: Ground truth category (`environment`, `financial`, etc.)

    **Example:**
    | text | sentiment | category |
    |------|-----------|----------|
    | "The market is booming today." | positive | financial |
    | "New climate regulations were introduced." | neutral | environment |

    ⚠️ Make sure to match the **exact column names**. If your columns are different, rename them in your notebook or Streamlit app.

    ---
    """)
