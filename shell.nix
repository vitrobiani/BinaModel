
# CUDA Training Environment for Bina Pipeline
# Usage: nix-shell (then training will auto-setup)

{ pkgs ? import (fetchTarball "https://github.com/nixos/nixpkgs/archive/nixos-unstable.tar.gz") {
    config.allowUnfree = true;
  }
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Clean Python (no packages baked in)
    python313

    # CUDA support
    linuxPackages.nvidia_x11
    cudaPackages.cudatoolkit

    # Build essentials
    gcc
    stdenv.cc.cc.lib
    zlib

    # OpenCV dependencies (X11/GUI libs)
    libx11
    libxext
    libxrender
    libxcb
    libGL
    libGLU
    glib

    # Utils
    tcl
    tk
  ];

  shellHook = ''
    # CUDA and OpenCV library paths
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libxcb}/lib:${pkgs.libx11}/lib:${pkgs.libxext}/lib:${pkgs.libxrender}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}

    # Tcl/Tk for matplotlib
    export TCL_LIBRARY="${pkgs.tcl}/lib/tcl${pkgs.tcl.version}"
    export TK_LIBRARY="${pkgs.tk}/lib/tk${pkgs.tk.version}"

    # CRITICAL: Disable NixOS system site-packages interference
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1

    # Create isolated venv
    export VENV_DIR="$PWD/.venv-cuda"
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating CUDA venv (fresh)..."
      rm -rf "$VENV_DIR"
      ${pkgs.python313}/bin/python3.13 -m venv "$VENV_DIR" --clear

      # Use venv python directly, not the wrapper
      export PATH="$VENV_DIR/bin:$PATH"

      # Force PYTHONPATH to only see venv packages
      export PYTHONPATH="$VENV_DIR/lib/python3.13/site-packages"

      "$VENV_DIR/bin/pip" install --upgrade pip -q

      echo "Installing PyTorch with CUDA 12.4..."
      "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

      echo "Installing other dependencies..."
      "$VENV_DIR/bin/pip" install ultralytics opencv-python tqdm pyyaml numpy -q
    else
      export PATH="$VENV_DIR/bin:$PATH"
      export PYTHONPATH="$VENV_DIR/lib/python3.13/site-packages"
    fi

    # Verify CUDA using venv python explicitly
    CUDA_OK=$("$VENV_DIR/bin/python" -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null || echo "NO")
    TORCH_VER=$("$VENV_DIR/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
    TORCH_FILE=$("$VENV_DIR/bin/python" -c "import torch; print(torch.__file__)" 2>/dev/null || echo "unknown")

    echo ""
    echo -e "\033[1;32m=== Bina CUDA Training Environment ===\033[0m"
    echo "Python: $("$VENV_DIR/bin/python" --version)"
    echo "PyTorch: $TORCH_VER"
    echo "Torch location: $TORCH_FILE"
    echo -e "CUDA available: \033[1;$([ "$CUDA_OK" = "YES" ] && echo "32" || echo "31")m$CUDA_OK\033[0m"
    if [ "$CUDA_OK" = "YES" ]; then
      echo "GPU: $("$VENV_DIR/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    fi
    echo ""
    echo "To train: python src/pipeline.py --phase train"
    echo ""
  '';
}
