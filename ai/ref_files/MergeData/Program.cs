using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using OfficeOpenXml;

// Cấu hình License cho EPPlus 

ExcelPackage.License.SetNonCommercialOrganization("My Noncommercial organization");
Random rand = new Random();

// Thiết lập đường dẫn thư mục theo đúng cấu hình của bạn
string targetFolder = @"C:\Users\Duy\Documents\MyProject\nienLuanCoSo\ml_dotnet_NLCS\All_files";

// Dùng Path.Combine để tạo đường dẫn tuyệt đối cho 3 file đầu vào (Excel) và 1 file xuất ra (CSV)
string pathThongKe = Path.Combine(targetFolder, "ThongKeByDuy.xlsx");
string pathHolland = Path.Combine(targetFolder, "Holland_ThongKe_Mapping.xlsx");
string pathDiemThi = Path.Combine(targetFolder, "DiemThi.xlsx");
string pathOutput = Path.Combine(targetFolder, "dataset.csv");

Console.WriteLine("1. Đang nạp Ma trận tri thức ngành từ file Excel (ThongKeByDuy.xlsx)...");
var dsNganh = new List<NhomNganhModel>();
using (var packageThongKe = new ExcelPackage(new FileInfo(pathThongKe)))
{
    var sheet = packageThongKe.Workbook.Worksheets[0];
    int rowCount = sheet.Dimension?.End.Row ?? 0;

    // Duyệt từ dòng 2 để bỏ qua Header
    for (int row = 2; row <= rowCount; row++)
    {
        // Cột 2: Label, Cột 3: ToHop, Cột 4: DiemTB, Cột 5: DiemMin
        string label = sheet.Cells[row, 2].Text;
        string toHopStr = sheet.Cells[row, 3].Text;
        string diemTbStr = sheet.Cells[row, 4].Text;
        string diemMinStr = sheet.Cells[row, 5].Text;

        if (string.IsNullOrWhiteSpace(label)) continue;

        dsNganh.Add(new NhomNganhModel
        {
            Label = label,
            ToHopHopLe = new HashSet<string>(toHopStr.Split(',').Select(x => x.Trim())),
            DiemTB = double.Parse(diemTbStr, CultureInfo.InvariantCulture),
            DiemMin = double.Parse(diemMinStr, CultureInfo.InvariantCulture)
        });
    }
}

Console.WriteLine("2. Đang nạp Ma trận Tính cách từ file Excel (Holland_ThongKe_Mapping.xlsx)...");
var dsHolland = new List<HollandModel>();
using (var packageHolland = new ExcelPackage(new FileInfo(pathHolland)))
{
    var sheet = packageHolland.Workbook.Worksheets[0];
    int rowCount = sheet.Dimension?.End.Row ?? 0;

    for (int row = 2; row <= rowCount; row++)
    {
        // Cột 1: TinhCach, Cột 2: DanhSachNganh
        string tinhCach = sheet.Cells[row, 1].Text;
        string danhSachNganh = sheet.Cells[row, 2].Text;

        if (string.IsNullOrWhiteSpace(tinhCach)) continue;

        dsHolland.Add(new HollandModel
        {
            TinhCach = tinhCach,
            DanhSachNhomNganh = new HashSet<string>(danhSachNganh.Split(';').Select(x => x.Trim()))
        });
    }
}

Console.WriteLine("3. Đang xử lý đối chiếu file DiemThi.xlsx và ghi ra file CSV...");
var csvOutputLines = new List<string>();

// Thêm Header cho file CSV đầu ra
csvOutputLines.Add("MaToHop,DiemToHop,NhomTinhCach,NhomNganh");

using (var packageDiemThi = new ExcelPackage(new FileInfo(pathDiemThi)))
{
    var sheet = packageDiemThi.Workbook.Worksheets[0];
    int rowCount = sheet.Dimension?.End.Row ?? 0;

    for (int row = 2; row <= rowCount; row++)
    {
        // Cột 2: MaToHop, Cột 3: DiemToHop
        string maToHop = sheet.Cells[row, 2].Text.Trim();
        string diemToHopStr = sheet.Cells[row, 3].Text;

        if (string.IsNullOrWhiteSpace(maToHop)) continue;
        if (!double.TryParse(diemToHopStr, NumberStyles.Any, CultureInfo.InvariantCulture, out double diemToHop)) continue;

        string nganhDuocChon = null;
        double minKhoangCach = double.MaxValue;

        // BƯỚC 1: XÉT ĐIỂM VÀ TỔ HỢP ĐỂ LỌC NGÀNH
        foreach (var nganh in dsNganh)
        {
            if (nganh.ToHopHopLe.Contains(maToHop) && diemToHop >= nganh.DiemMin)
            {
                double khoangCach = Math.Abs(diemToHop - nganh.DiemTB);
                if (khoangCach < minKhoangCach)
                {
                    minKhoangCach = khoangCach;
                    nganhDuocChon = nganh.Label;
                }
            }
        }

        // BƯỚC 2 & 3: TÌM TÍNH CÁCH (HOLLAND) VÀ LƯU VÀO LIST CHỜ XUẤT CSV
        if (!string.IsNullOrEmpty(nganhDuocChon))
        {
            var tinhCachPhuHop = dsHolland
                .Where(h => h.DanhSachNhomNganh.Contains(nganhDuocChon))
                .Select(h => h.TinhCach)
                .ToList();

            string tinhCachDuocChon = "";
            if (tinhCachPhuHop.Count > 0)
            {
                int index = rand.Next(tinhCachPhuHop.Count);
                tinhCachDuocChon = tinhCachPhuHop[index];
            }

            // Nối dữ liệu thành 1 dòng chuẩn CSV (các trường cách nhau bằng dấu phẩy)
            // Lưu ý: DiemToHop dùng ToString() với InvariantCulture để đảm bảo dấu thập phân là dấu chấm (.)
            string csvRow = $"{maToHop},{diemToHop.ToString(CultureInfo.InvariantCulture)},{tinhCachDuocChon},{nganhDuocChon}";
            csvOutputLines.Add(csvRow);
        }
    }
}

// BƯỚC 4: GHI TOÀN BỘ DỮ LIỆU RA FILE CSV (Định dạng UTF-8)
File.WriteAllLines(pathOutput, csvOutputLines, Encoding.UTF8);

Console.WriteLine("\nHOÀN TẤT!");
Console.WriteLine($"Đã đọc các file Excel từ thư mục: {targetFolder}");
Console.WriteLine($"Đã xuất file CSV tại: {pathOutput}");
Console.WriteLine($"Tổng số dòng dữ liệu hợp lệ sinh ra: {csvOutputLines.Count - 1}");

// --------------------------------------------------------
// PHẦN 2: KHAI BÁO CLASS (Bắt buộc để ở cuối file)
// --------------------------------------------------------

// Class lưu trữ cấu hình Nhóm Ngành
class NhomNganhModel
{
    public string Label { get; set; }
    public HashSet<string> ToHopHopLe { get; set; }
    public double DiemTB { get; set; }
    public double DiemMin { get; set; }
}

// Class lưu trữ cấu hình Holland
class HollandModel
{
    public string TinhCach { get; set; }
    public HashSet<string> DanhSachNhomNganh { get; set; }
}