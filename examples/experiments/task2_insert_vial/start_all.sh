#!/bin/bash
# Quick start script for remote deployment
# This script helps you start all services in the correct order

echo "=========================================="
echo "Remote Deployment Quick Start"
echo "=========================================="
echo ""

# Check if we're on robot machine or A800
echo "Which machine are you on?"
echo "  1) Robot Machine (Franka + RealSense)"
echo "  2) A800 Machine (GPU Server)"
echo ""
read -p "Enter choice [1-2]: " machine_choice

if [ "$machine_choice" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Robot Machine Setup"
    echo "=========================================="
    echo ""
    
    # Check if franka_server.py is running
    if pgrep -f "franka_server.py" > /dev/null; then
        echo "✓ franka_server.py is already running"
    else
        echo "Starting franka_server.py..."
        gnome-terminal -- bash -c "cd /home/szl/real_world_rl/serl_robot_infra/robot_servers && python franka_server.py --flask_url 0.0.0.0; exec bash"
        sleep 3
        echo "✓ franka_server.py started"
    fi
    
    # Check if env_server.py is running
    if pgrep -f "env_server.py" > /dev/null; then
        echo "✓ env_server.py is already running"
    else
        echo "Starting env_server.py..."
        gnome-terminal -- bash -c "cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana && ./run_env_server.sh; exec bash"
        sleep 3
        echo "✓ env_server.py started"
    fi
    
    echo ""
    echo "=========================================="
    echo "Robot Machine Ready!"
    echo "=========================================="
    echo ""
    echo "Next steps on A800:"
    echo "  1. Run: ./start_all.sh"
    echo "  2. Choose option 2 (A800 Machine)"
    echo "  3. Start learner first"
    echo "  4. Start actor second"
    echo ""
    
elif [ "$machine_choice" = "2" ]; then
    echo ""
    echo "=========================================="
    echo "A800 Machine Setup"
    echo "=========================================="
    echo ""
    
    # Get robot IP
    read -p "Enter Robot Machine IP (e.g., 192.168.1.100): " robot_ip
    
    # Update actor script with robot IP
    sed -i "s/ROBOT_IP=.*/ROBOT_IP=\"$robot_ip\"/" run_actor_conrft_remote.sh
    echo "✓ Updated run_actor_conrft_remote.sh with robot IP: $robot_ip"
    
    # Test connection
    echo ""
    echo "Testing Pyro5 connection to robot machine..."
    cd /home/szl/real_world_rl/serl_launcher
    python test_pyro_env.py --ip $robot_ip --port 9090
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Connection test passed!"
        echo ""
        echo "=========================================="
        echo "Ready to Start Training"
        echo "=========================================="
        echo ""
        echo "Choose what to start:"
        echo "  1) Learner only"
        echo "  2) Actor only"
        echo "  3) Both (in separate terminals)"
        echo ""
        read -p "Enter choice [1-3]: " start_choice
        
        if [ "$start_choice" = "1" ]; then
            echo "Starting learner..."
            cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
            ./run_learner_conrft_remote.sh
            
        elif [ "$start_choice" = "2" ]; then
            echo "Starting actor..."
            cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
            ./run_actor_conrft_remote.sh
            
        elif [ "$start_choice" = "3" ]; then
            echo "Starting learner in new terminal..."
            gnome-terminal -- bash -c "cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana && ./run_learner_conrft_remote.sh; exec bash"
            sleep 5
            echo "Starting actor in new terminal..."
            gnome-terminal -- bash -c "cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana && ./run_actor_conrft_remote.sh; exec bash"
            echo ""
            echo "✓ Both learner and actor started!"
            echo ""
            echo "Monitor training:"
            echo "  - Learner terminal: shows training progress"
            echo "  - Actor terminal: shows environment interaction"
            echo ""
        else
            echo "Invalid choice. Exiting."
            exit 1
        fi
    else
        echo ""
        echo "✗ Connection test failed!"
        echo ""
        echo "Please check:"
        echo "  1. Is env_server.py running on robot machine?"
        echo "  2. Is firewall allowing port 9090?"
        echo "  3. Is robot IP correct: $robot_ip"
        echo ""
        exit 1
    fi
else
    echo "Invalid choice. Exiting."
    exit 1
fi
