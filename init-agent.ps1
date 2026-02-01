if (!(Test-Path ".agent")) {
    mkdir .agent | Out-Null
}

if (!(Test-Path ".agent\skills")) {
    cmd /c mklink /J ".agent\skills" "$env:USERPROFILE\.agent\skills"
}

Write-Host "Antigravity skills linked."
