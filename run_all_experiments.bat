@echo off
REM Script pour lancer toutes les configurations de loss functions
REM Usage: run_all_experiments.bat

echo ==========================================
echo LANCEMENT DE TOUTES LES EXPERIENCES
echo ==========================================
echo.

REM Parametres communs
set COMMON_PARAMS=--use_memory --n_epoch 50 --patience 10 --lr 1e-4 --node_dim 200 --time_dim 200 --memory_dim 200 --message_dim 200 --n_runs 6 --use_wandb

REM Configuration 1: BCE (Baseline)
echo ==========================================
echo Configuration 1/4: BCE Baseline
echo ==========================================
python train_self_supervised.py --prefix tgn-bce %COMMON_PARAMS%
echo.
echo √ BCE termine
echo.

REM Configuration 2: Focal Loss
echo ==========================================
echo Configuration 2/4: Focal Loss
echo ==========================================
python train_self_supervised.py --prefix tgn-focal %COMMON_PARAMS% --use_focal_loss --focal_alpha 0.25 --focal_gamma 2.0
echo.
echo √ Focal Loss termine
echo.

REM Configuration 3: DCL Loss
echo ==========================================
echo Configuration 3/4: DCL Loss
echo ==========================================
python train_self_supervised.py --prefix tgn-dcl %COMMON_PARAMS% --use_dcl_loss --dcl_alpha 0.5 --dcl_temperature 0.07
echo.
echo √ DCL Loss termine
echo.

REM Configuration 4: Hybrid (Focal + DCL)
echo ==========================================
echo Configuration 4/4: Hybrid (Focal + DCL)
echo ==========================================
python train_self_supervised.py --prefix tgn-hybrid %COMMON_PARAMS% --use_focal_loss --use_dcl_loss --focal_alpha 0.25 --focal_gamma 2.0 --dcl_alpha 0.5 --dcl_temperature 0.07
echo.
echo √ Hybrid termine
echo.

echo ==========================================
echo √√ TOUTES LES EXPERIENCES SONT TERMINEES
echo ==========================================
echo.
echo Pour visualiser les resultats:
echo   python plot_loss_comparison.py
echo.
pause
