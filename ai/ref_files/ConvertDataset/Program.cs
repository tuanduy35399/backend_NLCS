using OfficeOpenXml;
using OfficeOpenXml.FormulaParsing.LexicalAnalysis;


ExcelPackage.License.SetNonCommercialOrganization("My Noncommercial organization");

string targetFolder = @"C:\Users\Duy\Documents\MyProject\ml_dotnet_NLCS\All_files";
string fileToHopPath = Path.Combine(targetFolder, "Cac_to_hop.xlsx");
string fileKetQuaThiPath = Path.Combine(targetFolder, "ketquathi2025.xlsx");
string dataset = Path.Combine(targetFolder, "dataset.xlsx");


try
{
    Console.WriteLine("Tai danh sach cac to hop mon");
    var dsToHop = DocDanhSachToHop(fileToHopPath);
    Console.WriteLine("Doc diem thi va tinh toan diem to hop");
    var ketQuaTinhToan = TinhDiemToHop(fileKetQuaThiPath, dsToHop);
    Console.WriteLine("Dang xuat ket qua ra file excel moi");
    XuatFileExcel(dataset, ketQuaTinhToan);
    Console.WriteLine($"Da xuat ket qua thanh cong, da luu tai: {dataset}");
}
catch (Exception ex)
{
    Console.WriteLine($"Da co loi: {ex.Message}");
}

//Dinh nghia cac ham xu ly

//Ham luu danh sach to hop tu file vao 1 dict (bo nho RAM)
static Dictionary<string, List<string>> DocDanhSachToHop(string fileToHopPath)
{
    var toHopDict = new Dictionary<string, List<string>>();
    using (var package = new ExcelPackage(new FileInfo(fileToHopPath)))
    {
        var worksheet = package.Workbook.Worksheets[0];
        int rowCount = worksheet.Dimension.Rows;
        for (int row = 2; row <= rowCount; row++)
        {
            string maToHop = worksheet.Cells[row, 1].Text.Trim();
            string chuoiMonHoc = worksheet.Cells[row, 2].Text;
            if (string.IsNullOrEmpty(maToHop) || string.IsNullOrEmpty(chuoiMonHoc)) continue;
            var danhSachMonHoc = chuoiMonHoc.Split(',')
                .Select(m => m.Trim().ToLower())
                .ToList();
            if (!toHopDict.ContainsKey(maToHop))
            {
                toHopDict.Add(maToHop, danhSachMonHoc);
            }
        }
        return toHopDict;
    }
}
//Ham tinh diem to hop 
static List<OutputRecord> TinhDiemToHop(string fileKetQuaThiPath, Dictionary<string, List<string>> toHopDict)
{
    var listResult = new List<OutputRecord>();
    using (var package = new ExcelPackage(new FileInfo(fileKetQuaThiPath)))
    {
        var worksheet = package.Workbook.Worksheets[0];
        int rowCount = worksheet.Dimension.Rows;
        int colCount = worksheet.Dimension.Columns;
        var headerMap = new Dictionary<string, int>();
        for (int col = 1; col <= colCount; col++)
        {
            string headerText = worksheet.Cells[1, col].Text.Trim().ToLower();
            if (!string.IsNullOrEmpty(headerText) && !headerMap.ContainsKey(headerText))
            {
                headerMap.Add(headerText, col);
            }
        }
        if (!headerMap.ContainsKey("sobaodanh"))
        {
            throw new Exception(@"File diem thi phai chua tieu de cot 'SOBAODANH' ");
        }
        int sbdCol = headerMap["sobaodanh"];
        for (int row = 2; row <= rowCount; row++)
        {
            string sbd = worksheet.Cells[row, sbdCol].Text.Trim();
            if (string.IsNullOrEmpty(sbd)) continue;
            var diemHocSinhDict = new Dictionary<string, double>();
            foreach (var pair in headerMap)
            {
                if (pair.Key == "sobaodanh") continue;

                string cellValue = worksheet.Cells[row, pair.Value].Text.Trim();
                if (double.TryParse(cellValue, out double diem))
                {
                    diemHocSinhDict[pair.Key] = diem;
                }
            }
            foreach (var toHop in toHopDict)
            {
                string maToHop = toHop.Key;
                List<string> cacMonYeuCau = toHop.Value;

                bool duDieuKien = true;
                double tongDiem = 0;

                foreach (var mon in cacMonYeuCau)
                {
                    if (diemHocSinhDict.ContainsKey(mon))
                    {
                        tongDiem += diemHocSinhDict[mon];
                    }
                    else
                    {
                        duDieuKien = false;
                        break;
                    }
                }

                if (duDieuKien)
                {
                    listResult.Add(new OutputRecord
                    {
                        SOBAODANH = sbd,
                        MaToHop = maToHop,
                        DiemToHop = Math.Round(tongDiem, 2)
                    });
                }
            }
        }
    }

    return listResult;
}

static void XuatFileExcel(string filePath, List<OutputRecord> data)
{
    var fileInfo = new FileInfo(filePath);
    if (fileInfo.Exists) fileInfo.Delete();

    using (var package = new ExcelPackage(fileInfo))
    {
        var worksheet = package.Workbook.Worksheets.Add("KetQuaToHop");

        worksheet.Cells[1, 1].Value = "SOBAODANH";
        worksheet.Cells[1, 2].Value = "Mã tổ hợp";
        worksheet.Cells[1, 3].Value = "Điểm tổ hợp";

        using (var range = worksheet.Cells[1, 1, 1, 3])
        {
            range.Style.Font.Bold = true;
        }

        int currentRow = 2;
        foreach (var item in data)
        {
            worksheet.Cells[currentRow, 1].Value = item.SOBAODANH;
            worksheet.Cells[currentRow, 2].Value = item.MaToHop;
            worksheet.Cells[currentRow, 3].Value = item.DiemToHop;
            currentRow++;
        }

        worksheet.Cells[worksheet.Dimension.Address].AutoFitColumns();
        package.Save();
    }
}

//Cái này là phải để ở cuối file!!!
public class OutputRecord
{
    public string? SOBAODANH { get; set; }
    public string? MaToHop { get; set; }
    public double? DiemToHop { get; set; }

}
