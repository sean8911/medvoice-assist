param(
    [Parameter(Mandatory = $true)]
    [string]$DocumentPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"

$documentFile = Get-Item -LiteralPath $DocumentPath
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Output directory already exists: $OutputDirectory"
}
$outputFolder = New-Item -ItemType Directory -Path $OutputDirectory
$pdfPath = Join-Path $outputFolder.FullName ($documentFile.BaseName + ".pdf")

$word = $null
$doc = $null
$powerPoint = $null
$presentation = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($documentFile.FullName, $false, $true, $false)

    $doc.ExportAsFixedFormat($pdfPath, 17)

    $pageCount = $doc.ComputeStatistics(2)

    $chapter4Range = $doc.Content.Duplicate
    $chapter4Range.Find.ClearFormatting()
    $chapter4Range.Find.Text = "CHAPTER 4"
    $chapter4Found = $chapter4Range.Find.Execute()
    $chapter4Page = if ($chapter4Found) { $chapter4Range.Information(3) } else { -1 }

    $chapter5Range = $doc.Content.Duplicate
    $chapter5Range.Find.ClearFormatting()
    $chapter5Range.Find.Text = "CHAPTER 5"
    $chapter5Found = $chapter5Range.Find.Execute()
    $chapter5Page = if ($chapter5Found) { $chapter5Range.Information(3) } else { -1 }
    if ($chapter4Page -lt 1 -or $chapter5Page -le $chapter4Page) {
        throw "Could not determine the Chapter 4 page range."
    }

    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $presentation = $powerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 900
    $presentation.PageSetup.SlideHeight = 900

    for ($page = $chapter4Page; $page -lt $chapter5Page; $page++) {
        $startRange = $doc.GoTo(1, 1, $page)
        $start = $startRange.Start
        if ($page -lt $pageCount) {
            $nextRange = $doc.GoTo(1, 1, $page + 1)
            $end = [Math]::Max($start, $nextRange.Start - 1)
        } else {
            $end = $doc.Content.End
        }

        $pageRange = $doc.Range($start, $end)
        $pageRange.CopyAsPicture()
        Start-Sleep -Milliseconds 120

        $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
        $shapeRange = $slide.Shapes.PasteSpecial(2)
        $shape = $shapeRange.Item(1)
        $shape.LockAspectRatio = -1

        $availableWidth = 860
        $availableHeight = 860
        $scale = [Math]::Min(
            $availableWidth / [double]$shape.Width,
            $availableHeight / [double]$shape.Height
        )
        $shape.Width = $shape.Width * $scale
        $shape.Height = $shape.Height * $scale
        $shape.Left = (900 - $shape.Width) / 2
        $shape.Top = (900 - $shape.Height) / 2

        $pngPath = Join-Path $outputFolder.FullName ("page-{0:D2}.png" -f $page)
        $slide.Export($pngPath, "PNG", 1800, 1800)
    }

    [pscustomobject]@{
        Document = $documentFile.FullName
        PDF = $pdfPath
        PageCount = $pageCount
        Chapter4Page = $chapter4Page
        Chapter5Page = $chapter5Page
        PNGDirectory = $outputFolder.FullName
    } | Format-List
}
finally {
    if ($presentation -ne $null) {
        $presentation.Close()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($powerPoint -ne $null) {
        $powerPoint.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
    if ($doc -ne $null) {
        $doc.Close($false)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
    if ($word -ne $null) {
        $word.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
