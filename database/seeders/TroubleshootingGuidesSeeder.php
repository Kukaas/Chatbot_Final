<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;

class TroubleshootingGuidesSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $troubleshootingData = [
            [
                'title' => 'How to Fix Slow Computer Performance',
                'content' => "If your computer is running slowly, follow these steps:\n\n" .
                            "1. Open Task Manager (Ctrl + Shift + Esc on Windows, or Command + Space, then type 'Activity Monitor' on macOS)\n" .
                            "2. Check the 'Processes' tab to identify resource-intensive applications\n" .
                            "3. End tasks for unnecessary processes\n" .
                            "4. Consider upgrading RAM or switching to an SSD if problems persist",
            ],
            [
                'title' => 'Troubleshooting Hardware Failures',
                'content' => "When experiencing hardware issues:\n\n" .
                            "1. Check all physical connections (power, display, peripherals)\n" .
                            "2. Restart the computer\n" .
                            "3. Run hardware diagnostics (F12 for Dell, F2 for HP during startup)\n" .
                            "4. Check device manager for error indicators",
            ],
            [
                'title' => 'Fixing USB Device Connection Problems',
                'content' => "If a USB device isn't working:\n\n" .
                            "1. Unplug and replug the device\n" .
                            "2. Try different USB ports\n" .
                            "3. Check device drivers in Device Manager/System Information\n" .
                            "4. Test the device on another computer if possible",
            ],
            [
                'title' => 'Resolving Battery Drain Issues',
                'content' => "To improve battery life:\n\n" .
                            "1. Check battery usage in settings\n" .
                            "2. Close background apps\n" .
                            "3. Disable unnecessary features (location, push notifications)\n" .
                            "4. Consider battery replacement if issues persist",
            ],
            [
                'title' => 'Dealing with Screen Damage',
                'content' => "When your screen is damaged:\n\n" .
                            "1. Back up your data immediately\n" .
                            "2. Contact manufacturer/certified repair center\n" .
                            "3. Use temporary protection measures\n" .
                            "4. Consider external display options",
            ],
            [
                'title' => 'Resolving Network Connectivity Problems',
                'content' => "If you're having network issues:\n\n" .
                            "1. Toggle Airplane Mode on/off\n" .
                            "2. Restart device and router\n" .
                            "3. Reset network settings\n" .
                            "4. Check for IP conflicts",
            ],
            [
                'title' => 'Preventing and Fixing Overheating Issues',
                'content' => "To address overheating:\n\n" .
                            "1. Clean vents and fans with compressed air\n" .
                            "2. Ensure proper airflow around device\n" .
                            "3. Monitor temperature with management tools\n" .
                            "4. Consider thermal paste replacement",
            ],
            [
                'title' => 'Managing Low Disk Space',
                'content' => "When running out of storage:\n\n" .
                            "1. Identify large files using disk analysis tools\n" .
                            "2. Clear temporary files and downloads\n" .
                            "3. Uninstall unused applications\n" .
                            "4. Consider cloud storage options",
            ],
            [
                'title' => 'Improving Slow Internet Connection',
                'content' => "To troubleshoot slow internet:\n\n" .
                            "1. Test speed at speedtest.net\n" .
                            "2. Restart modem and router\n" .
                            "3. Check for bandwidth-heavy applications\n" .
                            "4. Contact ISP if issues persist",
            ],
            [
                'title' => 'Fixing Failed System Updates',
                'content' => "When updates won't install:\n\n" .
                            "1. Verify internet connection stability\n" .
                            "2. Clear system drive space\n" .
                            "3. Download updates manually if needed\n" .
                            "4. Check for system file corruption",
            ],
            [
                'title' => 'WiFi Connection Troubleshooting',
                'content' => "If your WiFi is not working, follow these steps:\n\n" .
                            "1. Check WiFi is turned on:\n" .
                            "   - Look for WiFi icon in taskbar/menu bar\n" .
                            "   - Check physical WiFi switch on laptop\n" .
                            "   - Try Airplane Mode toggle\n\n" .
                            "2. Router checks:\n" .
                            "   - Verify router power and lights\n" .
                            "   - Restart router (unplug for 30 seconds)\n" .
                            "   - Check if others can connect\n\n" .
                            "3. Device-specific steps:\n" .
                            "   - Forget network and reconnect\n" .
                            "   - Reset network settings\n" .
                            "   - Update WiFi drivers\n\n" .
                            "4. Advanced troubleshooting:\n" .
                            "   - Check IP configuration\n" .
                            "   - Run network diagnostics\n" .
                            "   - Contact ISP if needed",
            ],
            [
                'title' => 'Common WiFi Problems and Solutions',
                'content' => "Common WiFi issues and their solutions:\n\n" .
                            "1. No WiFi Connection:\n" .
                            "   - Verify WiFi adapter is enabled\n" .
                            "   - Check for network availability\n" .
                            "   - Ensure correct password\n\n" .
                            "2. Weak Signal:\n" .
                            "   - Move closer to router\n" .
                            "   - Remove obstacles\n" .
                            "   - Consider WiFi extender\n\n" .
                            "3. Intermittent Connection:\n" .
                            "   - Update router firmware\n" .
                            "   - Change WiFi channel\n" .
                            "   - Check for interference\n\n" .
                            "4. Authentication Issues:\n" .
                            "   - Reset network credentials\n" .
                            "   - Check security settings\n" .
                            "   - Verify MAC filtering",
            ],
            [
                'title' => 'Network Connectivity Issues',
                'content' => "When experiencing network problems:\n\n" .
                            "1. Basic Network Checks:\n" .
                            "   - Verify network adapter status\n" .
                            "   - Test multiple websites\n" .
                            "   - Check other devices\n\n" .
                            "2. Connection Troubleshooting:\n" .
                            "   - Run Windows Network Diagnostics\n" .
                            "   - Reset TCP/IP stack\n" .
                            "   - Clear DNS cache\n\n" .
                            "3. Hardware Verification:\n" .
                            "   - Test with ethernet cable\n" .
                            "   - Check router settings\n" .
                            "   - Verify ISP status\n\n" .
                            "4. Advanced Solutions:\n" .
                            "   - Update network drivers\n" .
                            "   - Check firewall settings\n" .
                            "   - Reset network stack",
            ],
        ];

        foreach ($troubleshootingData as $data) {
            try {
                // Generate embedding for both title and content
                $embedding = $this->generateEmbedding($data['title'] . ' ' . $data['content']);

                DB::table('troubleshooting_guides')->insert([
                    'title' => $data['title'],
                    'content' => $data['content'],
                    'embedding' => $embedding,
                    'is_active' => true,
                    'created_at' => Carbon::now(),
                    'updated_at' => Carbon::now(),
                ]);
            } catch (\Exception $e) {
                echo "Error processing guide '{$data['title']}': {$e->getMessage()}\n";
            }
        }
    }

    private function generateEmbedding($text)
    {
        $response = Http::post('http://127.0.0.1:8000/generate-embedding', [
            'text' => $text
        ]);

        if ($response->successful()) {
            return json_encode($response->json()['embedding']);
        } else {
            throw new \Exception('Failed to generate embedding: ' . $response->body());
        }
    }
} 